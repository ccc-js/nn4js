const { CausalSelfAttention, MLP, Block, GPT } = require('../src/gpt.js');
const { Tensor } = require('../src/tensor.js');

describe('v0.6: 自注意力機制', () => {
  describe('CausalSelfAttention', () => {
    test('CausalSelfAttention 建立', () => {
      const attn = new CausalSelfAttention(16, 4);
      expect(attn.n_head).toBe(4);
      expect(attn.head_dim).toBe(4);
    });

    test('CausalSelfAttention forward', () => {
      const attn = new CausalSelfAttention(16, 4);
      const x = new Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]], [], true);
      const [out, kv] = attn.__call__(x);
      expect(out.data.shape).toEqual([1, 1, 16]);
      expect(kv.length).toBe(2);
    });

    test('CausalSelfAttention backward', () => {
      const attn = new CausalSelfAttention(16, 4);
      const x = new Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]], [], true);
      const [out, kv] = attn.__call__(x);
      out.backward();
      expect(x.grad.data[0]).not.toBeNaN();
    });

    test('CausalSelfAttention with KV cache', () => {
      const attn = new CausalSelfAttention(16, 4);
      const x1 = new Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]], [], true);
      const [out1, kv1] = attn.__call__(x1);

      const x2 = new Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]], [], true);
      const [out2, kv2] = attn.__call__(x2, kv1);
      expect(out2.data.shape).toEqual([1, 1, 16]);
    });
  });

  describe('MLP', () => {
    test('MLP forward', () => {
      const mlp = new MLP(16);
      const x = new Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]], [], true);
      const out = mlp.__call__(x);
      expect(out.data.shape).toEqual([1, 1, 16]);
    });
  });

  describe('Block', () => {
    test('Block forward', () => {
      const block = new Block(16, 4);
      const x = new Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]], [], true);
      const [out, cache] = block.__call__(x);
      expect(out.data.shape).toEqual([1, 1, 16]);
    });
  });

  describe('GPT', () => {
    test('GPT 建立', () => {
      const gpt = new GPT(50, 16, 1, 16, 4);
      const params = gpt.parameters();
      expect(params.length).toBeGreaterThan(0);
    });

    test('GPT forward', () => {
      const gpt = new GPT(50, 16, 1, 16, 4);
      const idx = new Tensor([[1, 2, 3, 4]]);
      const [logits, caches] = gpt.__call__(idx);
      expect(logits.data.shape).toEqual([1, 4, 50]);
    });

    test('GPT forward with kv_caches', () => {
      const gpt = new GPT(50, 16, 1, 16, 4);
      const idx1 = new Tensor([[1, 2]]);
      const [_, caches1] = gpt.__call__(idx1);

      const idx2 = new Tensor([[3]]);
      const [logits, caches2] = gpt.__call__(idx2, caches1);
      expect(logits.data.shape).toEqual([1, 1, 50]);
    });
  });
});