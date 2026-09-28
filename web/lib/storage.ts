// Small things remembered on this device, like the key, so a friend only types it once.
// Private windows and some browsers refuse storage, so every read and write is guarded.

const PREFIX = "sympai.";

export function load(name: string, fallback = ""): string {
  try {
    return window.localStorage.getItem(PREFIX + name) ?? fallback;
  } catch {
    return fallback;
  }
}

export function save(name: string, value: string) {
  try {
    window.localStorage.setItem(PREFIX + name, value);
  } catch {
    /* storage is off, the page still works for this visit */
  }
}

export function remove(name: string) {
  try {
    window.localStorage.removeItem(PREFIX + name);
  } catch {
    /* nothing to do */
  }
}
