const { Tensor } = require('./tensor.js');
const { Adam } = require('./nn.js');
const { GPT } = require('./gpt.js');

function train_model(model, optimizer, docs, uchars, BOS, block_size, num_steps = 100) {
  const params = model.parameters();
  const charToIdx = {};
  for (let i = 0; i < uchars.length; i++) {
    charToIdx[uchars[i]] = i;
  }

  for (let step = 0; step < num_steps; step++) {
    const doc = docs[step % docs.length];
    if (!doc || doc.length === 0) continue;
    const tokens = [BOS];
    for (const ch of doc) {
      if (charToIdx[ch] !== undefined) {
        tokens.push(charToIdx[ch]);
      }
    }
    tokens.push(BOS);
    const n = Math.min(block_size, tokens.length - 1);
    if (n <= 0) continue;

    const xData = tokens.slice(0, n);
    const yData = tokens.slice(1, n + 1);

    const x = new Tensor(toNdarray(xData, [1, n]), [], true);
    const y = new Tensor(toNdarray(yData, [1, n]), [], false);

    optimizer.zero_grad();
    const [logits, _] = model.__call__(x, null);

    const loss = logits.cross_entropy(y);
    loss.backward();

    const max_norm = 1.0;
    let total_norm = 0;
    for (const p of params) {
      let sum = 0;
      for (let i = 0; i < p.grad.data.length; i++) {
        sum += p.grad.data[i] * p.grad.data[i];
      }
      total_norm += sum;
    }
    total_norm = Math.sqrt(total_norm);

    if (total_norm > max_norm) {
      const clip_coef = max_norm / (total_norm + 1e-6);
      for (const p of params) {
        for (let i = 0; i < p.grad.data.length; i++) {
          p.grad.data[i] *= clip_coef;
        }
      }
    }

    optimizer.step();
    optimizer.lr = 0.01 * (1 - step / num_steps);

    if (step % 10 === 0 || step === num_steps - 1) {
      console.log(`step ${step + 1}/${num_steps} | loss ${loss.data.data[0].toFixed(4)}`);
    }
  }

  return model;
}

function generate_samples(model, uchars, BOS, vocab_size, block_size, num_samples = 5, temperature = 0.5) {
  console.log('\n--- inference ---');
  const results = [];

  for (let sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    let current_token = BOS;
    const sample = [];
    let kv_caches = null;

    for (let pos_id = 0; pos_id < block_size; pos_id++) {
      const x = new Tensor(toNdarray([current_token], [1, 1]), [], false);
      const [logits, kv_caches_new] = model.__call__(x, kv_caches);
      kv_caches = kv_caches_new;

      const last_logits = logits.data.data;

      const maxLogit = Math.max(...last_logits);
      let sumExp = 0;
      const exps = new Float32Array(vocab_size);
      for (let v = 0; v < vocab_size; v++) {
        exps[v] = Math.exp(last_logits[v] / temperature - maxLogit / temperature);
        sumExp += exps[v];
      }

      const probs = exps.map(e => e / sumExp);

      let r = Math.random();
      let next_token = 0;
      let cumsum = 0;
      for (let v = 0; v < vocab_size; v++) {
        cumsum += probs[v];
        if (r <= cumsum) {
          next_token = v;
          break;
        }
      }

      if (next_token === BOS) break;
      sample.push(uchars[next_token]);
      current_token = next_token;
    }

    const generated_name = sample.join('');
    results.push(generated_name);
    console.log(`sample ${sample_idx + 1}: ${generated_name}`);
  }

  return results;
}

const ndarray = require('ndarray');

function toNdarray(data, shape) {
  return ndarray(new Float32Array(data), shape);
}

module.exports = { train_model, generate_samples };