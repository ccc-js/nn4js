const { Block, GPT } = require('../src/gpt.js');
const { Tensor } = require('../src/tensor.js');

describe('v0.7: Transformer 區塊與 GPT 模型', () => {
  describe('Block', () => {
    test('Block forward', () => {
      const block = new Block(16, 4);
      const x = new Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]], [], true);
      const [out, cache] = block.__call__(x);
      expect(out.data.shape).toEqual([1, 1, 16]);
    });

    test('Block backward', () => {
      const block = new Block(16, 4);
      const x = new Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]], [], true);
      const [out, cache] = block.__call__(x);
      out.backward();
      expect(x.grad.data[0]).not.toBeNaN();
    });

    test('Block with KV cache', () => {
      const block = new Block(16, 4);
      const x1 = new Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]], [], true);
      const [_, cache1] = block.__call__(x1);

      const x2 = new Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]], [], true);
      const [out2, cache2] = block.__call__(x2, cache1);
      expect(out2.data.shape).toEqual([1, 1, 16]);
    });
  });

  describe('GPT', () => {
    test('GPT 建立', () => {
      const gpt = new GPT(50, 16, 1, 16, 4);
      const params = gpt.parameters();
      expect(params.length).toBeGreaterThan(0);
    });

    test('GPT parameters 數量', () => {
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

    test('GPT backward', () => {
      const gpt = new GPT(50, 16, 1, 16, 4);
      const idx = new Tensor([[1, 2, 3]], [], true);
      const [logits, _] = gpt.__call__(idx);
      logits.backward();
      expect(logits.grad.data[0]).not.toBeNaN();
    });

    test('GPT 多層', () => {
      const gpt = new GPT(50, 16, 2, 16, 4);
      const idx = new Tensor([[1, 2]]);
      const [logits, _] = gpt.__call__(idx);
      expect(logits.data.shape).toEqual([1, 2, 50]);
    });
  });
});