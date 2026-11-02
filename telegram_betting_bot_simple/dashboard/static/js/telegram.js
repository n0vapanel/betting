// Glue for running inside Telegram as a Mini App. Outside Telegram every helper here does nothing.
//
// Telegram's script (telegram-web-app.js, loaded in index.html) gives window.Telegram.WebApp; it has
// initData only when Telegram opened the page. The app then signs in with that initData, takes
// Telegram's colours, uses Telegram's own bottom button for a screen's main action and its Back
// button for closing sheets and going back, and taps the phone lightly when a pick is toggled.
// Every call is wrapped: an old Telegram app missing a method must never break the page.

const webApp = window.Telegram?.WebApp;

/** True when Telegram opened the page (there is signed initData to sign in with). */
export const inTelegram = Boolean(webApp?.initData);

export function initData() {
  return inTelegram ? webApp.initData : "";
}

/** The deep-link parameter Telegram passes with t.me/<bot>/<app>?startapp=..., if any. */
export function startParam() {
  return inTelegram ? webApp.initDataUnsafe?.start_param || "" : "";
}

/** The Telegram user's first name (unverified, for greetings only). */
export function firstName() {
  return inTelegram ? webApp.initDataUnsafe?.user?.first_name || "" : "";
}

function safely(action) {
  try {
    return action();
  } catch {
    return undefined;
  }
}

function supports(version) {
  return Boolean(safely(() => webApp.isVersionAtLeast(version)));
}

// --- theme and layout ----------------------------------------------------------------------------

function applyTheme() {
  const root = document.documentElement;
  root.dataset.theme = webApp.colorScheme === "light" ? "light" : "dark";
  // Telegram sets --tg-theme-* itself; mirror the header and bottom bar so they blend in
  const params = webApp.themeParams || {};
  const bar = params.secondary_bg_color || params.bg_color;
  if (bar) {
    safely(() => webApp.setHeaderColor(bar));
    safely(() => webApp.setBackgroundColor(bar));
    if (supports("7.10")) safely(() => webApp.setBottomBarColor(bar));
  }
}

function applySafeAreas() {
  const root = document.documentElement.style;
  const safe = webApp.safeAreaInset || {};
  const content = webApp.contentSafeAreaInset || {};
  // Full screen puts Telegram's own controls over the top of the page: keep content below them
  root.setProperty("--tg-top", `${(safe.top || 0) + (content.top || 0)}px`);
  root.setProperty("--tg-bottom", `${(safe.bottom || 0) + (content.bottom || 0)}px`);
}

/** The Telegram user's id as Telegram's page data says (unverified: only for keeping one person's cached copy apart). */
export function userId() {
  const id = inTelegram ? webApp.initDataUnsafe?.user?.id : null;
  return id === undefined || id === null ? "" : String(id);
}

/**
 * Open the app full height and take Telegram's colours. Safe to call outside Telegram. It does not call
 * ready(): Telegram keeps its own loading screen up until ready(), so app.js calls ready() once the first
 * screen is drawn (from the cache when there is one), not before and not after the network refresh.
 */
export function initTelegram() {
  if (!inTelegram) return;
  document.documentElement.classList.add("tg");
  applyTheme();
  applySafeAreas();
  safely(() => webApp.expand());
  safely(() => webApp.onEvent("themeChanged", applyTheme));
  safely(() => webApp.onEvent("safeAreaChanged", applySafeAreas));
  safely(() => webApp.onEvent("contentSafeAreaChanged", applySafeAreas));
}

let readyAt = null;

/**
 * Tell Telegram the first screen is drawn, so it swaps its loading screen for the app. Once per visit;
 * returns the performance.now() time it was first called (null outside Telegram).
 */
export function ready() {
  if (!inTelegram) return null;
  if (readyAt === null) {
    readyAt = performance.now();
    safely(() => webApp.ready());
  }
  return readyAt;
}

/** Swipe-down-to-close off (on screens with charts or sideways scrolling) or back on. */
export function setVerticalSwipes(enabled) {
  if (!inTelegram || !supports("7.7")) return;
  safely(() => (enabled ? webApp.enableVerticalSwipes() : webApp.disableVerticalSwipes()));
}

// --- haptics: a light tap must never break the page -----------------------------------------------

export const haptic = {
  /** A pick added or removed. */
  select() {
    if (inTelegram) safely(() => webApp.HapticFeedback.selectionChanged());
  },
  tap(style = "light") {
    if (inTelegram) safely(() => webApp.HapticFeedback.impactOccurred(style));
  },
  /** "success" | "warning" | "error", e.g. when an analysis finishes or fails. */
  notify(kind = "success") {
    if (inTelegram) safely(() => webApp.HapticFeedback.notificationOccurred(kind));
  },
};

// --- Telegram's bottom buttons ---------------------------------------------------------------------

let mainHandler = null;
let secondaryHandler = null;

/**
 * Show Telegram's main bottom button for a screen's main action, or hide it with null.
 *   setMainButton({ text: "Analyse with AI", onClick, disabled, progress, analyse })
 * Returns true when Telegram shows it (the page can then skip drawing its own button). ``analyse`` marks
 * the slip's Analyse, which app.js uses to step the slip bar's own Analyse aside.
 */
export function setMainButton(config) {
  if (!inTelegram) return false;
  const button = webApp.MainButton;
  if (mainHandler) safely(() => button.offClick(mainHandler));
  mainHandler = null;
  if (!config) {
    safely(() => button.hide());
    return false;
  }
  mainHandler = () => config.onClick?.();
  // True only once Telegram took it: an old client without the button leaves the page's own buttons in place
  return Boolean(safely(() => {
    button.setParams({ text: config.text, is_active: !config.disabled, is_visible: true });
    button.onClick(mainHandler);
    if (config.progress) button.showProgress(false);
    else button.hideProgress();
    return true;
  }));
}

/** Telegram's second bottom button (Bot API 7.10+), e.g. "Clear"; null hides it. */
export function setSecondaryButton(config) {
  if (!inTelegram || !webApp.SecondaryButton || !supports("7.10")) return false;
  const button = webApp.SecondaryButton;
  if (secondaryHandler) safely(() => button.offClick(secondaryHandler));
  secondaryHandler = null;
  if (!config) {
    safely(() => button.hide());
    return false;
  }
  secondaryHandler = () => config.onClick?.();
  safely(() => {
    button.setParams({ text: config.text, is_active: !config.disabled, is_visible: true, position: "left" });
    button.onClick(secondaryHandler);
  });
  return true;
}

// --- Telegram's Back button --------------------------------------------------------------------------

let backHandler = null;

/** Show Telegram's Back button with a handler, or hide it with null. */
export function setBackButton(handler) {
  if (!inTelegram) return;
  const button = webApp.BackButton;
  if (backHandler) safely(() => button.offClick(backHandler));
  backHandler = handler || null;
  safely(() => {
    if (backHandler) {
      button.onClick(backHandler);
      button.show();
    } else {
      button.hide();
    }
  });
}

// --- links -----------------------------------------------------------------------------------------

/** Open an outside link: Telegram's in-app browser inside Telegram, a new tab elsewhere. */
export function openLink(url) {
  if (!/^https?:\/\//i.test(url || "")) return;
  if (inTelegram && safely(() => (webApp.openLink(url), true))) return;
  window.open(url, "_blank", "noopener,noreferrer");
}

/** Close the Mini App (e.g. after a refused sign-in). */
export function closeApp() {
  if (inTelegram) safely(() => webApp.close());
}
