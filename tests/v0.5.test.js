const { Module, Linear, Embedding, RMSNorm, Adam } = require('../src/nn.js');
const { Tensor } = require('../src/tensor.js');

describe('v0.5: 正規化與優化器', () => {
  describe('RMSNorm', () => {
    test('RMSNorm 正規化', () => {
      const layer = new RMSNorm(4);
      const x = new Tensor([[1, 2, 3, 4]], [], true);
      const out = layer.__call__(x);
      expect(out.data.shape).toEqual([1, 4]);
      expect(out.data.data[0]).not.toBeNaN();
    });

    test('RMSNorm backward', () => {
      const layer = new RMSNorm(4);
      const x = new Tensor([[1, 2, 3, 4]], [], true);
      const out = layer.__call__(x);
      out.backward();
      expect(x.grad.data[0]).not.toBeNaN();
    });
  });

  describe('Adam', () => {
    test('Adam 建立', () => {
      const layer = new Linear(3, 2);
      const params = layer.parameters();
      const opt = new Adam(params, lr = 0.01);
      expect(opt.params.length).toBe(1);
      expect(opt.t).toBe(0);
    });

    test('Adam step 更新參數', () => {
      const layer = new Linear(3, 2, true);
      const params = layer.parameters();
      const opt = new Adam(params, lr = 0.1);

      const x = new Tensor([[1, 2, 3]], [], true);
      const out = layer.__call__(x);
      out.backward();

      const oldWeight = layer.weight.data.data.slice();
      opt.step();
      const newWeight = layer.weight.data.data;

      let changed = false;
      for (let i = 0; i < oldWeight.length; i++) {
        if (oldWeight[i] !== newWeight[i]) {
          changed = true;
          break;
        }
      }
      expect(changed).toBe(true);
    });

    test('Adam zero_grad', () => {
      const layer = new Linear(3, 2);
      const params = layer.parameters();
      const opt = new Adam(params, lr = 0.01);

      const x = new Tensor([[1, 2, 3]], [], true);
      const out = layer.__call__(x);
      out.backward();

      opt.zero_grad();
      expect(layer.weight.grad.data[0]).toBe(0);
    });
  });
});