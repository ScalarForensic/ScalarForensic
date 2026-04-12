# Using the vector visualization as a KDE wallpaper or screensaver

The ScalarForensic web server exposes a self-contained visualization page at
`/viz` that renders the indexed embedding space with no further server calls
after the initial load.  It is designed to fill any screen and looks
particularly good on OLED displays (pure black centre, no chrome).

---

## The `/viz` endpoint

`http://localhost:8080/viz` returns a single HTML file with:

- the full animated canvas (rotating point cloud, traverser flythroughs,
  spark connectors)
- the point-cloud data embedded as inline JSON — no polling, no websocket
- `viz.js` inlined — the file is completely self-contained once loaded

You can also generate a **static copy** written to disk at server startup
(useful for wallpaper plugins that prefer a `file://` URL or when the server
is not always running — see below).

---

## Option A — KDE Plasma "Web Page" wallpaper (live, server must be running)

This is the simplest setup and always shows the latest indexed collection.

1. Right-click the desktop → **Configure Desktop and Wallpaper…**
2. Wallpaper type → **Web Page** (install `plasma-wallpaper-webpage` if missing:
   `sudo dnf install plasma-wallpaper-webpage` or the equivalent for your distro)
3. URL → `http://localhost:8080/viz`
4. Apply

The wallpaper updates automatically whenever the server restarts with a new
collection.

---

## Option B — Static exported file (works without a running server)

Set the environment variable `SFN_VIZ_EXPORT_PATH` before starting the server.
The server writes the rendered HTML to that path on every startup.

```bash
# in your .env file or shell profile
SFN_VIZ_EXPORT_PATH=~/.local/share/sfn/viz.html
```

The directory is created automatically.  After starting `sfn-web` once,
the file exists and can be used with a `file://` URL at any time,
even when the server is not running.

Point the **Web Page** wallpaper plugin at:

```
file:///home/YOUR_USER/.local/share/sfn/viz.html
```

---

## Option C — Screensaver via xscreensaver (any desktop)

`xscreensaver` can run an arbitrary command as a screensaver.

1. Install: `sudo dnf install xscreensaver` (or your distro's equivalent)

2. Create `~/.xscreensaver` (or edit the existing one) and add:

   ```
   programs: \
     sfn-viz -root \n\
   ```

3. Create the wrapper script `~/bin/sfn-viz`:

   ```bash
   #!/bin/bash
   # Renders the viz in the root window (xscreensaver -root mode).
   # Requires xwinwrap: https://github.com/ujjwal96/xwinwrap
   xwinwrap -g $(xdpyinfo | grep dimensions | awk '{print $2}') \
            -fs -sp -ov -nf -- \
            chromium --app=http://localhost:8080/viz \
                     --start-fullscreen \
                     --noerrdialogs \
                     --disable-infobars \
                     --window-position=0,0 \
                     -WID %WID
   ```

   Make it executable: `chmod +x ~/bin/sfn-viz`

4. Start xscreensaver: add `xscreensaver -no-splash &` to your autostart.

> **Simpler alternative** — if you just want a screensaver without xwinwrap,
> create a KDE autostart script that watches `xprintidle` and opens a
> fullscreen browser window:
>
> ```bash
> #!/bin/bash
> IDLE_MS=300000   # 5 minutes
> while true; do
>   idle=$(xprintidle 2>/dev/null || echo 0)
>   if [ "$idle" -gt "$IDLE_MS" ]; then
>     chromium --app=http://localhost:8080/viz --start-fullscreen \
>              --noerrdialogs &
>     PID=$!
>     # Wait until the user moves the mouse, then kill it
>     while [ "$(xprintidle)" -gt 1000 ]; do sleep 2; done
>     kill $PID 2>/dev/null
>   fi
>   sleep 10
> done
> ```
>
> Save as `~/.local/bin/sfn-screensaver.sh`, make executable, and add it to
> **System Settings → Autostart**.

---

## Option D — KDE Plasma scripted screensaver (no xscreensaver)

KDE Plasma 6 does not ship traditional screensavers, but you can trigger
a fullscreen window via the **Power Management** idle action:

1. **System Settings → Power Management → Energy Saving**
2. Under "After a period of inactivity": enable **Run script**
3. Script:
   ```bash
   chromium --app=http://localhost:8080/viz --start-fullscreen \
            --noerrdialogs --disable-infobars
   ```
4. To close it when the session resumes, add a **resume script**:
   ```bash
   pkill -f "chromium.*localhost:8080/viz"
   ```

---

## Choosing a browser

Any Chromium-based browser works (`chromium`, `google-chrome`, `brave`,
`microsoft-edge`).  Firefox supports a similar kiosk mode:

```bash
firefox --kiosk http://localhost:8080/viz
```

---

## Notes

- The visualization renders at whatever resolution the window occupies —
  no configuration needed for HiDPI or ultra-wide displays.
- If `SFN_VIZ_MAX_POINTS=0` is set, the point cloud is empty and
  `/viz` will show only the animated background gradient.
- The page registers mouse drag and scroll handlers unconditionally, but
  they only have an effect if the user actively interacts with the page;
  no extra setup is required for typical wallpaper or screensaver use.
