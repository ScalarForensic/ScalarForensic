// sfn() Alpine component assembler
// The component is defined in topical parts under static/js/, each pushing an
// object fragment onto window.__sfnParts. See index.html for script load order.
// Fragments are merged with property descriptors — NOT Object.assign — so the
// computed getters in computed.js are copied as getters instead of being
// evaluated once at merge time.
function sfn() {
  const component = {};
  for (const part of window.__sfnParts || []) {
    Object.defineProperties(component, Object.getOwnPropertyDescriptors(part));
  }
  return component;
}
