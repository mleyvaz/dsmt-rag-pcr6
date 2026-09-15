import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { AutoTokenizer, AutoModelForSequenceClassification, env } from '@huggingface/transformers';

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {};
  for (let i = 0; i < args.length; i += 2) out[args[i].replace(/^--/, '')] = args[i + 1];
  if (!out.data || !out.output) throw new Error('Usage: node run_nli_scoring.mjs --data FILE --output FILE [--cache DIR] [--batch 16] [--limit 0]');
  return out;
}

function normalize(text) {
  return String(text).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim().replace(/\s+/g, ' ');
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map(x => Math.exp(x - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / total);
}

async function loadJsonl(file, limit) {
  const rows = [];
  const stream = fs.createReadStream(file, { encoding: 'utf8' });
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
  for await (const line of rl) {
    if (line.trim()) rows.push(JSON.parse(line));
    if (limit && rows.length >= limit) break;
  }
  return rows;
}

const args = parseArgs();
const batchSize = Number(args.batch || 16);
const limit = Number(args.limit || 0);
const cacheDir = args.cache || path.resolve(path.dirname(args.output), 'hf_cache');
fs.mkdirSync(path.dirname(args.output), { recursive: true });
fs.mkdirSync(cacheDir, { recursive: true });
env.cacheDir = cacheDir.replaceAll('\\', '/');

const modelId = 'Xenova/distilbert-base-uncased-mnli';
const tokenizer = await AutoTokenizer.from_pretrained(modelId);
const model = await AutoModelForSequenceClassification.from_pretrained(modelId, { dtype: 'q8' });
const items = await loadJsonl(args.data, limit);

const tasks = [];
for (let itemId = 0; itemId < items.length; itemId++) {
  const item = items[itemId];
  const candidateMap = new Map();
  for (const doc of item.documents) {
    const key = normalize(doc.answer);
    if (key && key !== 'unknown' && !candidateMap.has(key)) candidateMap.set(key, String(doc.answer));
  }
  const candidates = [...candidateMap.entries()].map(([key, text]) => ({ key, text }));
  for (let docIndex = 0; docIndex < item.documents.length; docIndex++) {
    const doc = item.documents[docIndex];
    for (const candidate of candidates) {
      const hypothesis = `The answer to the question "${item.question}" is "${candidate.text}".`;
      tasks.push({ item_id: itemId, doc_index: docIndex, candidate: candidate.key, premise: String(doc.text), hypothesis });
    }
  }
}

const outputs = [];
for (let start = 0; start < tasks.length; start += batchSize) {
  const batch = tasks.slice(start, start + batchSize);
  const premises = batch.map(x => x.premise);
  const hypotheses = batch.map(x => x.hypothesis);
  const inputs = tokenizer(premises, { text_pair: hypotheses, padding: true, truncation: true, max_length: 256 });
  const result = await model(inputs);
  const logits = await result.logits.tolist();
  for (let i = 0; i < batch.length; i++) {
    const probs = softmax(logits[i]);
    outputs.push({
      item_id: batch[i].item_id,
      doc_index: batch[i].doc_index,
      candidate: batch[i].candidate,
      entailment: probs[0],
      neutral: probs[1],
      contradiction: probs[2],
    });
  }
  if (start % (batchSize * 25) === 0) process.stderr.write(`NLI ${Math.min(start + batchSize, tasks.length)}/${tasks.length}\n`);
}

const lines = outputs.map(x => JSON.stringify(x)).join('\n') + '\n';
fs.writeFileSync(args.output, lines, 'utf8');
const meta = {
  model: modelId,
  dtype: 'q8',
  max_length: 256,
  items: items.length,
  pairs: outputs.length,
  label_order: ['ENTAILMENT', 'NEUTRAL', 'CONTRADICTION'],
};
fs.writeFileSync(`${args.output}.meta.json`, JSON.stringify(meta, null, 2), 'utf8');
console.log(JSON.stringify(meta));
