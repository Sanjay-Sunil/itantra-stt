// IndicTrans2 text pre/post-processing, ported from IndicTransToolkit.processor (IndicProcessor, inference mode)
// with the pieces of indic-nlp-library (normalizer, trivial tokenizer/detokenizer, Unicode transliterator) and
// sacremoses (English punctuation normalizer, tokenizer, detokenizer) that it calls.

export const FLORES = { hi: "hin_Deva", gu: "guj_Gujr", mr: "mar_Deva", kn: "kan_Knda", ml: "mal_Mlym",
  ta: "tam_Taml", te: "tel_Telu", or: "ory_Orya", bn: "ben_Beng", en: "eng_Latn" };
const ISO = Object.fromEntries(Object.entries(FLORES).map(([k, v]) => [v, k]));

// ---------------------------------------------------------------- transliteration (indicnlp UnicodeIndicTransliterator)
const SCRIPT_BASE = { pa: 0x0a00, gu: 0x0a80, or: 0x0b00, ta: 0x0b80, te: 0x0c00, kn: 0x0c80, ml: 0x0d00,
  hi: 0x0900, mr: 0x0900, bn: 0x0980 };

function tamilOffset(o) {
  if (o >= 0x15 && o <= 0x28 && o !== 0x1c && !((o - 0x15) % 5 === 0 || (o - 0x15) % 5 === 4))
    o = 0x15 + 5 * Math.floor((o - 0x15) / 5);
  if (o === 0x2b || o === 0x2c || o === 0x2d) o = 0x2a;
  if (o === 0x36) o = 0x37;
  return o;
}

export function transliterate(text, from, to) {
  if (!(from in SCRIPT_BASE) || !(to in SCRIPT_BASE)) return text;
  let out = "";
  for (const c of text) {
    const cp = c.codePointAt(0);
    let o = cp - SCRIPT_BASE[from];
    if (o >= 0 && o <= 0x6f && c !== "\u0964" && c !== "\u0965") {
      if (to === "ta") o = tamilOffset(o);
      out += String.fromCodePoint(SCRIPT_BASE[to] + o);
    } else out += c;
  }
  return out;
}

// ---------------------------------------------------------------- indicnlp normalizers (default options)
const rep = (t, pairs) => pairs.reduce((s, [a, b]) => s.replaceAll(a, b), t);

function baseNormalize(t) {
  t = rep(t, [["\ufeff", ""], ["\ufffe", ""], ["\u2060", ""], ["\u00ad", ""], ["\u200b", " "], ["\u00a0", " "],
    ["\u200c", ""], ["\u200d", ""]]);
  // NormalizerI._normalize_punctuations
  return rep(t, [["\ufeff", ""], ["„", '"'], ["“", '"'], ["”", '"'], ["–", "-"], ["—", " - "], ["´", "'"], ["‘", "'"],
    ["‚", "'"], ["’", "'"], ["''", '"'], ["´´", '"'], ["…", "..."]]);
}

// Upstream defines NUKTA as the 6-character text "\\u093C" (escaped backslash), so decomposed nukta
// characters come out as that literal text; kept for parity with the Python engine.
// The upstream visarga regexes are written with escaped backslashes and never match; they are omitted here.
const NORMALIZERS = {
  hi: (t) => rep(baseNormalize(t), [["\u0972", "\u090f"], ["\u0929", "\u0928\\u093C"], ["\u0931", "\u0930\\u093C"],
    ["\u0934", "\u0933\\u093C"], ["\u0958", "\u0915\\u093C"], ["\u0959", "\u0916\\u093C"], ["\u095a", "\u0917\\u093C"],
    ["\u095b", "\u091c\\u093C"], ["\u095c", "\u0921\\u093C"], ["\u095d", "\u0922\\u093C"], ["\u095e", "\u092b\\u093C"],
    ["\u095f", "\u092f\\u093C"], ["|", "\u0964"]]),
  gu: (t) => rep(baseNormalize(t), [["\u0ae4", "\u0964"], ["\u0ae5", "\u0965"]]),
  or: (t) => rep(baseNormalize(t), [["\u0b05\u0b3e", "\u0b06"], ["\u0b0f\u0b57", "\u0b10"], ["\u0b13\u0b57", "\u0b14"],
    ["\u0b5c", "\u0b21\\u0B3C"], ["\u0b5d", "\u0b22\\u0B3C"], ["\u0b64", "\u0964"], ["\u0b65", "\u0965"],
    ["\u0b7c", "\u0964"], ["\u0b35", "\u0b2c"], ["\u0b47\u0b56", "\u0b58"], ["\u0b47\u0b3e", "\u0b4b"],
    ["\u0b47\u0b57", "\u0b4c"]]),
  bn: (t) => rep(baseNormalize(t), [["\u09dc", "\u09a1\\u09BC"], ["\u09dd", "\u09a2\\u09BC"], ["\u09df", "\u09af\\u09BC"],
    ["\u09e4", "\u0964"], ["\u09e5", "\u0965"], ["|", "\u0964"], ["\u09f7", "\u0964"], ["\u09c7\u09be", "\u09cb"],
    ["\u09c7\u09d7", "\u09cc"]]),
  ta: (t) => rep(baseNormalize(t), [["\u0be4", "\u0964"], ["\u0be5", "\u0965"], ["\u0b92\u0bd7", "\u0b94"],
    ["\u0bc6\u0bbe", "\u0bca"], ["\u0bc7\u0bbe", "\u0bcb"], ["\u0bc6\u0bd7", "\u0bcc"]]),
  te: (t) => rep(baseNormalize(t), [["\u0c64", "\u0964"], ["\u0c65", "\u0965"], ["\u0c46\u0c56", "\u0c48"]]),
  kn: (t) => rep(baseNormalize(t), [["\u0ce4", "\u0964"], ["\u0ce5", "\u0965"], ["\u0cbf\u0cd5", "\u0cc0"],
    ["\u0cc6\u0cd5", "\u0cc7"], ["\u0cc6\u0cd6", "\u0cc8"], ["\u0cc6\u0cc2", "\u0cca"], ["\u0cca\u0cd5", "\u0ccb"]]),
  ml: (t) => {
    t = rep(t, [["\u0d23\u0d4d\u200d", "\u0d7a"], ["\u0d28\u0d4d\u200d", "\u0d7b"], ["\u0d30\u0d4d\u200d", "\u0d7c"],
      ["\u0d32\u0d4d\u200d", "\u0d7d"], ["\u0d33\u0d4d\u200d", "\u0d7e"], ["\u0d15\u0d4d\u200d", "\u0d7f"]]);
    return rep(baseNormalize(t), [["\u0d64", "\u0964"], ["\u0d65", "\u0965"], ["\u0d46\u0d3e", "\u0d4a"],
      ["\u0d47\u0d3e", "\u0d4b"], ["\u0d46\u0d57", "\u0d4c"], ["\u0d57", "\u0d4c"]]);
  },
};
NORMALIZERS.mr = NORMALIZERS.hi;

// ---------------------------------------------------------------- indicnlp trivial tokenizer / detokenizer
// string.punctuation inside a Python character class: "\\]" escapes the bracket, so backslash itself is not included
const PUNCT = "!\"#$%&'()*+,\\-./:;<=>?@\\[\\]^_`{|}~";
const TRIV_TOK = new RegExp(`([${PUNCT}\u0964\u0965\uaaf1\uaaf0\uabeb\uabec\uabed\uabee\uabef\u1c7e\u1c7f])`, "gu");
const NUM_SEQ = /([0-9]+ [,.:/] )+[0-9]+/gu;

function trivialTokenize(text) {
  let s = text.replaceAll("\t", " ").replace(TRIV_TOK, " $1 ").replace(/[ ]+/g, " ").replace(/^ +| +$/g, "");
  s = s.replace(NUM_SEQ, (m) => m.replaceAll(" ", ""));
  return s.split(" ");
}

function trivialDetokenize(text) {
  let s = text;
  // upstream only removes spaces in number sequences that are preceded by other text (a quirk kept for parity)
  const parts = [];
  let prev = 0;
  for (const m of s.matchAll(NUM_SEQ)) {
    if (m.index > prev) {
      parts.push(s.slice(prev, m.index), m[0].replaceAll(" ", ""));
      prev = m.index + m[0].length;
    }
  }
  parts.push(s.slice(prev));
  s = parts.join("");
  s = s.replace(/[ ]([-/\\])[ ]/gu, "$1");
  s = s.replace(/[ ]([!%)\]},.:;>?\u0964\u0965])/gu, "$1");
  s = s.replace(/([#$(\[{<@])[ ]/gu, "$1");
  for (const punc of "'\"`") {
    let cnt = 0, out = "";
    for (const c of s) {
      if (c === punc) { out += cnt % 2 === 0 ? "@RA" : "@LA"; cnt++; } else out += c;
    }
    s = out.replaceAll("@RA ", punc).replaceAll(" @LA", punc).replaceAll("@RA", punc).replaceAll("@LA", punc);
  }
  return s;
}

// ---------------------------------------------------------------- sacremoses (English)
const ALNUM = "\\p{L}\\p{M}\\p{Nd}", ALPHA = "\\p{L}\\p{M}", ISN = "\\p{N}";
const MOSES_NORM = [
  [/\r/g, ""], [/\(/g, " ("], [/\)/g, ") "], [/ +/g, " "], [/\) ([.!:?;,])/g, ")$1"], [/\( /g, "("], [/ \)/g, ")"],
  [/(\d) %/gu, "$1%"], [/ :/g, ":"], [/ ;/g, ";"], [/`/g, "'"], [/''/g, ' " '], [/„/g, '"'], [/“/g, '"'], [/”/g, '"'],
  [/–/g, "-"], [/—/g, " - "], [/ +/g, " "], [/´/g, "'"], [/([a-zA-Z])‘([a-zA-Z])/g, "$1'$2"],
  [/([a-zA-Z])’([a-zA-Z])/g, "$1'$2"], [/‘/g, "'"], [/‚/g, "'"], [/’/g, "'"], [/''/g, '"'], [/´´/g, '"'],
  [/…/g, "..."], [/\u00a0«\u00a0/g, '"'], [/«\u00a0/g, '"'], [/«/g, '"'], [/\u00a0»\u00a0/g, '"'], [/\u00a0»/g, '"'],
  [/»/g, '"'], [/\u00a0%/g, "%"], [/nº\u00a0/g, "nº "], [/\u00a0:/g, ":"], [/\u00a0ºC/g, " ºC"], [/\u00a0cm/g, " cm"],
  [/\u00a0\?/g, "?"], [/\u00a0!/g, "!"], [/\u00a0;/g, ";"], [/,\u00a0/g, ", "], [/ +/g, " "], [/"([,.]+)/g, '$1"'],
  [/(\d)\u00a0(\d)/gu, "$1.$2"],
];
const mosesNormalize = (t) => MOSES_NORM.reduce((s, [re, r]) => s.replace(re, r), t).trim();

const NBP = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
  "W", "X", "Y", "Z", "Adj", "Adm", "Adv", "Asst", "Bart", "Bldg", "Brig", "Bros", "Capt", "Cmdr", "Col", "Comdr", "Con",
  "Corp", "Cpl", "DR", "Dr", "Drs", "Ens", "Gen", "Gov", "Hon", "Hr", "Hosp", "Insp", "Lt", "MM", "MR", "MRS", "MS",
  "Maj", "Messrs", "Mlle", "Mme", "Mr", "Mrs", "Ms", "Msgr", "Op", "Ord", "Pfc", "Ph", "Prof", "Pvt", "Rep", "Reps",
  "Res", "Rev", "Rt", "Sen", "Sens", "Sfc", "Sgt", "Sr", "St", "Supt", "Surg", "v", "vs", "i.e", "rev", "e.g", "Rs",
  "No #NUMERIC_ONLY#", "Nos", "Art #NUMERIC_ONLY#", "Nr", "pp #NUMERIC_ONLY#", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul",
  "Aug", "Sep", "Oct", "Nov", "Dec"];
const NUMERIC_ONLY = ["No", "Art", "pp"];
const re_ = (s) => new RegExp(s, "gu");
const MOSES_TOK = {
  padNotAlnum: re_(`([^${ALNUM}\\s.'\`,\\-])`),
  comma1: re_(`([^${ISN}])[,]`), comma2: re_(`[,]([^${ISN}])`), comma3: re_(`([${ISN}])[,]$`),
  en1: re_(`([^${ALPHA}])[']([^${ALPHA}])`), en2: re_(`([^${ALPHA}${ISN}])[']([${ALPHA}])`),
  en3: re_(`([${ALPHA}])[']([^${ALPHA}])`), en4: re_(`([${ALPHA}])[']([${ALPHA}])`), en5: re_(`([${ISN}])[']([s])`),
};
// Python re.sub never re-scans its own output; JS replace is the same, so each rule is a single pass.
function mosesTokenize(text) {
  const T = MOSES_TOK;
  let t = text.replace(/\s+/gu, " ").replace(/[\u0000-\u001f]/g, "").trim();
  t = t.replace(T.padNotAlnum, " $1 ");
  t = t.replace(/\.{2,}/g, (m, off) => " " + "DOT".repeat(m.length) + "MULTI" + (off + m.length < t.length ? " " : ""));
  t = t.replace(T.comma1, "$1 , ").replace(T.comma2, " , $1").replace(T.comma3, "$1 , ");
  t = t.replace(T.en1, "$1 ' $2").replace(T.en2, "$1 ' $2").replace(T.en3, "$1 ' $2").replace(T.en4, "$1 '$2")
    .replace(T.en5, "$1 '$2");
  const toks = t.split(/\s+/).filter(Boolean);
  for (let i = 0; i < toks.length; i++) {
    const m = /^(\S+)\.$/u.exec(toks[i]);
    if (!m) continue;
    const prefix = m[1];
    if ((prefix.includes(".") && /\p{L}/u.test(prefix)) || (NBP.includes(prefix) && !NUMERIC_ONLY.includes(prefix)) ||
      (i !== toks.length - 1 && toks[i + 1] && /^\p{Ll}$/u.test(toks[i + 1][0]))) continue;
    if (NUMERIC_ONLY.includes(prefix) && i + 1 < toks.length && /^[0-9]+/.test(toks[i + 1])) continue;
    toks[i] = prefix + " .";
  }
  t = toks.join(" ").replace(/\s+/gu, " ").trim().replace(/\.' ?$/, " . ' ");
  return t.replace(/(?:DOT)+MULTI/g, (m) => ".".repeat((m.length - 5) / 3)).split(/\s+/).filter(Boolean);
}

const CURRENCY = /^[$¢£¤¥֏؋৲৳৻૱௹฿៛₠-₿꠸﷼﹩＄￠￡￥￦\p{Sc}]/u;
function mosesDetokenize(tokens) {
  let text = ` ${tokens.join(" ")} `.replace(/ @-@ /g, "-");
  text = text.replace(/&#124;/g, "|").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'").replace(/&#91;/g, "[").replace(/&#93;/g, "]").replace(/&amp;/g, "&");
  const quotes = {};
  let pre = " ", out = "";
  const toks = text.split(/\s+/).filter(Boolean);
  toks.forEach((tok, i) => {
    if (CURRENCY.test(tok) || /^[(\[{¿¡]+$/u.test(tok)) { out += pre + tok; pre = ""; }
    else if (/^[,.?!:;\\%}\])]+$/u.test(tok)) { out += tok; pre = " "; }
    else if (i > 0 && /^['][\p{L}]/u.test(tok)) { out += tok; pre = " "; }
    else if (/^['"„“`]+$/u.test(tok)) {
      const q = /^[„“”]+$/u.test(tok) ? '"' : tok;
      quotes[q] = quotes[q] || 0;
      if (quotes[q] % 2 === 0) {
        if (tok === "'" && i > 0 && /[s]$/.test(toks[i - 1])) { out += tok; pre = " "; }
        else { out += pre + tok; pre = ""; quotes[q] += 1; }
      } else { out += tok; pre = " "; quotes[q] += 1; }
    } else { out += pre + tok; pre = " "; }
  });
  return out.replace(/ {2,}/g, " ").trim();
}

// ---------------------------------------------------------------- IndicProcessor
const PUNC_REPLACEMENTS = [
  [/\r/g, ""], [/\(\s*/gu, "("], [/\s*\)/gu, ")"], [/\s:\s?/gu, ":"], [/\s;\s?/gu, ";"], [/[`´‘‚’]/g, "'"],
  [/[„“”«»]/g, '"'], [/[–—]/g, "-"], [/\.\.\./g, "..."], [/ %/g, "%"], [/nº /g, "nº "], [/ ºC/g, " ºC"],
  [/ [?!;]/g, (m) => m.trim()], [/, /g, ", "],
];
function puncNorm(t) {
  for (const [re, r] of PUNC_REPLACEMENTS) t = t.replace(re, r);
  return t.replace(/[ ]{2,}/g, " ").replace(/\) ([.!:?;,])/g, ")$1").replace(/(\p{Nd}) %/gu, "$1%")
    .replace(/"([,.]+)/g, '$1"').replace(/(\p{Nd})\u00a0(\p{Nd})/gu, "$1.$2").trim();
}

const DIGIT_ZERO = [0x09e6, 0x0ae6, 0x0ce6, 0x0966, 0x0660, 0xabf0, 0x0b66, 0x0a66, 0x1c50, 0x06f0, 0x0c66];
function normalizeDigits(t) {
  return t.replace(/\p{Nd}/gu, (c) => {
    const cp = c.codePointAt(0);
    for (const z of DIGIT_ZERO) if (cp >= z && cp <= z + 9) return String(cp - z);
    return c;
  });
}

const W = "[\\p{L}\\p{M}\\p{Nd}_]";
// Unicode word boundary (JS \\b is ASCII-only even with the u flag; Python's regex module is Unicode-aware)
const B = `(?:(?<=${W})(?!${W})|(?<!${W})(?=${W}))`;
const PLACEHOLDER_PATTERNS = [
  ["email", /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}/g],
  ["url", new RegExp(`${B}(?<![\\p{L}\\p{M}\\p{Nd}_/.])(?:(?:https?|ftp)://)?(?:(?:[\\p{L}\\p{M}\\p{Nd}_-]+\\.)+(?!\\.))(?:[\\p{L}\\p{M}\\p{Nd}_/\\-?#&=%.]+)+(?!\\.${W}+)${B}`, "gu")],
  ["num", /(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-/.,:']\d+[-/.,:'+]\d+(?:\.\d+)?|\d+[-/.:'+]\d+(?:\.\d+)?)/g],
  ["other", new RegExp(`[A-Za-z0-9]*[#|@]${W}+`, "gu")],
];
const INDIC_FAILURE_CASES = ["آی ڈی ", "ꯑꯥꯏꯗꯤ", "आईडी", "आई . डी . ", "आई . डी .", "आई. डी. ", "आई. डी.", "आय. डी. ",
  "आय. डी.", "आय . डी . ", "आय . डी .आइ . डी . ", "आइ . डी .", "आइ. डी. ", "आइ. डी.", "ऐटि", "آئی ڈی ", "ᱟᱭᱰᱤ ᱾", "आयडी",
  "ऐडि", "आइडि", "ᱟᱭᱰᱤ"];

function wrapWithPlaceholders(text) {
  const map = new Map();
  let serial = 1;
  for (const [kind, re] of PLACEHOLDER_PATTERNS) {
    const matches = [...new Set([...text.matchAll(re)].map((m) => m[0]))];
    for (const m of matches) {
      if (kind === "url" && m.replaceAll(".", "").length < 4) continue;
      if (kind === "num" && m.replaceAll(" ", "").replaceAll(".", "").replaceAll(":", "").length < 4) continue;
      const n = serial;
      for (const id of ["ID", "id"])
        for (const f of [`<${id}${n}>`, `< ${id}${n} >`, `[${id}${n}]`, `[ ${id}${n} ]`, `[${id} ${n}]`, `<${id}${n}]`,
          `< ${id}${n}]`, `<${id}${n} ]`]) map.set(f, m);
      for (const c of INDIC_FAILURE_CASES)
        for (const f of [`<${c}${n}>`, `< ${c}${n} >`, `< ${c} ${n} >`, `<${c} ${n}]`, `< ${c} ${n} ]`, `[${c}${n}]`,
          `[${c} ${n}]`, `[ ${c}${n} ]`, `[ ${c} ${n} ]`, `${c} ${n}`, `${c}${n}`]) map.set(f, m);
      text = text.replaceAll(m, `<ID${n}>`);
      serial++;
    }
  }
  text = text.replace(/\s+/gu, " ").replaceAll(">/", ">").replaceAll("]/", "]");
  return { text, map };
}

/** → { text: model input "src_tag tgt_tag tokens…", map: placeholder map for postprocess } */
export function preprocess(sent, src, tgt) {
  const srcTag = FLORES[src], iso = src;
  sent = puncNorm(sent);
  sent = normalizeDigits(sent);
  const w = wrapWithPlaceholders(sent);
  sent = w.text;
  let out;
  if (iso === "en") out = mosesTokenize(mosesNormalize(sent.trim())).join(" ");
  else {
    const normed = NORMALIZERS[iso](sent.trim());
    out = transliterate(trivialTokenize(normed).join(" "), iso, "hi").replaceAll(" ् ", "्");
  }
  return { text: `${srcTag} ${FLORES[tgt]} ${out.trim()}`, map: w.map };
}

export function postprocess(sent, tgt, map) {
  const lang = FLORES[tgt];
  if (lang.startsWith("ory_")) sent = sent.replaceAll("ଯ଼", "ୟ");
  for (const [k, v] of map) sent = sent.replaceAll(k, v);
  if (lang === "eng_Latn") return mosesDetokenize(sent.split(" "));
  return trivialDetokenize(transliterate(sent, "hi", ISO[lang]));
}
