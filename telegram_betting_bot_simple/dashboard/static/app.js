// Entry point. The app is plain ES modules under /static/js (no build step): js/app.js is the shell
// (router, header, tab bar, bet slip, sign-in) and documents how screens plug in; js/views/ holds the screens.
import { boot } from "./js/app.js";

boot();
