const { Tensor, cat } = require('./tensor.js');
const { Module, Linear, Embedding, RMSNorm } = require('./nn.js');
const ndarray = require('ndarray');

class CausalSelfAttention extends Module {
  constructor(n_embd, n_head) {
    super();
    this.n_head = n_head;
    this.head_dim = n_embd / n_head;

    this.wq = new Linear(n_embd, n_embd);
    this.wk = new Linear(n_embd, n_embd);
    this.wv = new Linear(n_embd, n_embd);
    this.wo = new Linear(n_embd, n_embd);
  }

  __call__(x, kv_cache = null) {
    const B = x.data.shape[0];
    const T = x.data.shape[1];
    const C = x.data.shape[2];

    let q = this.wq.__call__(x);
    q = q.reshape(B, T, this.n_head, this.head_dim);
    q = q.transpose(0, 2, 1, 3);

    let k = this.wk.__call__(x);
    k = k.reshape(B, T, this.n_head, this.head_dim);
    k = k.transpose(0, 2, 1, 3);

    let v = this.wv.__call__(x);
    v = v.reshape(B, T, this.n_head, this.head_dim);
    v = v.transpose(0, 2, 1, 3);

    if (kv_cache !== null) {
      const [past_k, past_v] = kv_cache;
      k = cat([past_k, k], 2);
      v = cat([past_v, v], 2);
    }

    const T_k = k.data.shape[2];

    let attn_logits = q.matmul(k.transpose(-2, -1));
    attn_logits = attn_logits.mul(1.0 / Math.sqrt(this.head_dim));

    if (T > 1) {
      const maskData = new Float32Array(T * T_k);
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T_k; j++) {
          maskData[i * T_k + j] = i < j ? 1 : 0;
        }
      }
      const mask = new Tensor(ndarray(maskData, [T, T_k]));
      attn_logits = attn_logits.masked_fill(mask, -1e9);
    }

    const attn_weights = attn_logits.softmax(-1);
    let out = attn_weights.matmul(v);

    out = out.transpose(0, 2, 1, 3);
    out = out.reshape(B, T, C);

    out = this.wo.__call__(out);

    return [out, [k, v]];
  }
}

class MLP extends Module {
  constructor(n_embd) {
    super();
    this.fc1 = new Linear(n_embd, 4 * n_embd);
    this.fc2 = new Linear(4 * n_embd, n_embd);
  }

  __call__(x) {
    let out = this.fc1.__call__(x);
    out = out.relu();
    out = this.fc2.__call__(out);
    return out;
  }
}

class Block extends Module {
  constructor(n_embd, n_head) {
    super();
    this.attn = new CausalSelfAttention(n_embd, n_head);
    this.mlp = new MLP(n_embd);
    this.ln1 = new RMSNorm(n_embd);
    this.ln2 = new RMSNorm(n_embd);
  }

  __call__(x, kv_cache = null) {
    const [attn_out, new_cache] = this.attn.__call__(this.ln1.__call__(x), kv_cache);
    let out = x.add(attn_out);
    out = out.add(this.mlp.__call__(this.ln2.__call__(out)));
    return [out, new_cache];
  }
}

class GPT extends Module {
  constructor(vocab_size, block_size, n_layer = 1, n_embd = 16, n_head = 4) {
    super();
    this.block_size = block_size;
    this.wte = new Embedding(vocab_size, n_embd);
    this.wpe = new Embedding(block_size, n_embd);
    this.blocks = [];
    for (let i = 0; i < n_layer; i++) {
      this.blocks.push(new Block(n_embd, n_head));
    }
    this.ln_f = new RMSNorm(n_embd);
    this.lm_head = new Linear(n_embd, vocab_size);
  }

  __call__(idx, kv_caches = null) {
    const B = idx.data.shape[0];
    const T = idx.data.shape[1];

    let past_len = 0;
    if (kv_caches !== null) {
      past_len = kv_caches[0][0].data.shape[2];
    }

    const posData = new Float32Array(B * T);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        posData[b * T + t] = past_len + t;
      }
    }

    const posTensor = new Tensor(ndarray(posData, [B, T]));
    const tok_emb = this.wte.__call__(idx);
    const pos_emb = this.wpe.__call__(posTensor);
    let x = tok_emb.add(pos_emb);

    let new_caches = [];
    for (let i = 0; i < this.blocks.length; i++) {
      let layer_cache = null;
      if (kv_caches !== null) {
        layer_cache = kv_caches[i];
      }
      const [block_out, new_cache] = this.blocks[i].__call__(x, layer_cache);
      x = block_out;
      new_caches.push(new_cache);
    }

    x = this.ln_f.__call__(x);
    const logits = this.lm_head.__call__(x);

    return [logits, new_caches];
  }
}

module.exports = { CausalSelfAttention, MLP, Block, GPT };
