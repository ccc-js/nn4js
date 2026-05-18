const { Module, Linear, Embedding } = require('../src/nn.js');
const { Tensor } = require('../src/tensor.js');

describe('v0.4: 網路層', () => {
  describe('Module', () => {
    test('parameters 收集直接參數', () => {
      const linear = new Linear(3, 2, false);
      const params = linear.parameters();
      expect(params.length).toBe(1);
      expect(params[0]).toBe(linear.weight);
    });

    test('parameters 收集 bias 參數', () => {
      const linear = new Linear(3, 2, true);
      const params = linear.parameters();
      expect(params.length).toBe(2);
    });
  });

  describe('Linear', () => {
    test('Linear forward 無 bias', () => {
      const layer = new Linear(3, 2, false);
      const x = new Tensor([[1, 2, 3]], [], true);
      const out = layer.__call__(x);
      expect(out.data.shape).toEqual([1, 2]);
    });

    test('Linear forward 有 bias', () => {
      const layer = new Linear(3, 2, true);
      const x = new Tensor([[1, 2, 3]], [], true);
      const out = layer.__call__(x);
      expect(out.data.shape).toEqual([1, 2]);
    });

    test('Linear backward', () => {
      const layer = new Linear(3, 2, true);
      const x = new Tensor([[1, 2, 3]], [], true);
      const out = layer.__call__(x);
      out.backward();
      expect(x.grad.data[0]).not.toBeNaN();
    });
  });

  describe('Embedding', () => {
    test('Embedding 查表', () => {
      const layer = new Embedding(10, 4);
      const indices = new Tensor([[1, 2, 3]]);
      const out = layer.__call__(indices);
      expect(out.data.shape).toEqual([1, 3, 4]);
    });

    test('Embedding backward', () => {
      const layer = new Embedding(10, 4);
      const indices = new Tensor([[1, 2]], [], true);
      const out = layer.__call__(indices);
      out.backward();
      expect(layer.weight.grad.data[0]).not.toBeNaN();
    });
  });
});