"""Tests for HashCache in embedder.py.

Covers:
  - get_or_hash_both: cache hit on repeated call (no re-hash)
  - get_or_hash_both: cache miss when mtime changes
  - get_or_hash_both: cache miss when file size changes
  - get_or_hash_both: entries persist across HashCache instances (survive close/reopen)
  - _cache_key: uses resolved absolute path (relative paths map to the same key as absolute)
"""

import os
from pathlib import Path
from unittest import mock

import pytest

from scalar_forensic.embedder import HashCache


@pytest.fixture()
def tmp(tmp_path: Path) -> Path:
    return tmp_path


def _bump_mtime(path: Path) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_mtime_ns + 1_000_000_000, st.st_mtime_ns + 1_000_000_000))


class TestHashCacheGetOrHashBoth:
    def test_cached_on_second_call(self, tmp: Path) -> None:
        db = tmp / "cache.db"
        f = tmp / "file.bin"
        f.write_bytes(b"hello")

        cache = HashCache(db)
        try:
            with mock.patch(
                "scalar_forensic.embedder.hash_file_both",
                return_value=("sha-first", "md5-first"),
            ) as m:
                sha1, md1, hit1 = cache.get_or_hash_both(f)
                sha2, md2, hit2 = cache.get_or_hash_both(f)

            assert (sha1, md1, hit1) == ("sha-first", "md5-first", False)
            assert (sha2, md2, hit2) == ("sha-first", "md5-first", True)
            assert m.call_count == 1
        finally:
            cache.close()

    def test_invalidated_on_mtime_change(self, tmp: Path) -> None:
        db = tmp / "cache.db"
        f = tmp / "file.bin"
        f.write_bytes(b"hello")

        cache = HashCache(db)
        try:
            with mock.patch(
                "scalar_forensic.embedder.hash_file_both",
                side_effect=[("sha-a", "md5-a"), ("sha-b", "md5-b")],
            ) as m:
                first = cache.get_or_hash_both(f)
                _bump_mtime(f)
                second = cache.get_or_hash_both(f)

            assert first == ("sha-a", "md5-a", False)
            assert second == ("sha-b", "md5-b", False)
            assert m.call_count == 2
        finally:
            cache.close()

    def test_invalidated_on_size_change(self, tmp: Path) -> None:
        db = tmp / "cache.db"
        f = tmp / "file.bin"
        f.write_bytes(b"hello")

        cache = HashCache(db)
        try:
            with mock.patch(
                "scalar_forensic.embedder.hash_file_both",
                side_effect=[("sha-a", "md5-a"), ("sha-b", "md5-b")],
            ) as m:
                first = cache.get_or_hash_both(f)
                f.write_bytes(b"hello world")  # different size
                second = cache.get_or_hash_both(f)

            assert first == ("sha-a", "md5-a", False)
            assert second == ("sha-b", "md5-b", False)
            assert m.call_count == 2
        finally:
            cache.close()

    def test_persists_across_instances(self, tmp: Path) -> None:
        db = tmp / "cache.db"
        f = tmp / "file.bin"
        f.write_bytes(b"persistent")

        cache1 = HashCache(db)
        with mock.patch(
            "scalar_forensic.embedder.hash_file_both",
            return_value=("sha-p", "md5-p"),
        ) as m:
            cache1.get_or_hash_both(f)
        cache1.close()
        assert m.call_count == 1

        cache2 = HashCache(db)
        try:
            with mock.patch("scalar_forensic.embedder.hash_file_both") as m2:
                sha, md5, hit = cache2.get_or_hash_both(f)
            assert (sha, md5, hit) == ("sha-p", "md5-p", True)
            m2.assert_not_called()
        finally:
            cache2.close()


class TestHashCacheKey:
    def test_relative_and_absolute_resolve_to_same_key(self, tmp: Path) -> None:
        f = tmp / "file.bin"
        f.write_bytes(b"x")

        # Temporarily change cwd so a relative path points to the same file.
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)
            relative = Path("file.bin")
            absolute = f.resolve()
            assert HashCache._cache_key(relative) == HashCache._cache_key(absolute)
        finally:
            os.chdir(old_cwd)
