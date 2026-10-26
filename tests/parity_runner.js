// Runs the browser fallback in Node for tests/test_parity.py.
const fs = require("fs");
const L = require("../docs/app/lexical.js");
const data = JSON.parse(fs.readFileSync(__dirname + "/../docs/app/data/index.json", "utf8"));
const idx = L.load(data);
const queries = JSON.parse(process.argv[2]);
const out = queries.map((q) => {
  const s = idx.bm25.scores(q);
  const order = s.map((x, i) => [x, i]).sort((a, b) => (b[0] - a[0]) || (a[1] - b[1])).slice(0, 10);
  const top4 = order.slice(0, 4).map((r) => data.passages[r[1]]);
  return { q, rank: order.map((r) => r[1]), scores: order.map((r) => r[0]), meta: L.classify(q),
           tokens: L.tokens(q), extractive: L.extractive(q, top4, idx.idfText).text, safety: L.safetyCheck(q).kind };
});
process.stdout.write(JSON.stringify(out));
