const { Tensor, cat } = require('../src/tensor.js');

describe('v0.3: 張量形狀操作與激活函數', () => {
  describe('transpose', () => {
    test('2D 矩陣轉置', () => {
      const t = new Tensor([[1, 2, 3], [4, 5, 6]]);
      const result = t.transpose(0, 1);
      expect(result.data.get(0, 0)).toBe(1);
      expect(result.data.get(0, 1)).toBe(4);
      expect(result.data.get(1, 0)).toBe(2);
      expect(result.data.get(1, 1)).toBe(5);
      expect(result.data.get(2, 0)).toBe(3);
      expect(result.data.get(2, 1)).toBe(6);
    });

    test('transpose 反向傳播', () => {
      const t = new Tensor([[1, 2], [3, 4]], [], true);
      const result = t.transpose(0, 1);
      result.backward();
      expect(t.grad.data[0]).toBe(1);
      expect(t.grad.data[1]).toBe(1);
      expect(t.grad.data[2]).toBe(1);
      expect(t.grad.data[3]).toBe(1);
    });
  });

  describe('reshape', () => {
    test('1D 轉 2D', () => {
      const t = new Tensor([1, 2, 3, 4]);
      const result = t.reshape(2, 2);
      expect(result.data.get(0, 0)).toBe(1);
      expect(result.data.get(0, 1)).toBe(2);
      expect(result.data.get(1, 0)).toBe(3);
      expect(result.data.get(1, 1)).toBe(4);
    });

    test('reshape 反向傳播', () => {
      const t = new Tensor([1, 2, 3, 4], [], true);
      const result = t.reshape(2, 2);
      result.backward();
      expect(t.grad.data[0]).toBe(1);
      expect(t.grad.data[1]).toBe(1);
    });
  });

  describe('relu', () => {
    test('ReLU 激活', () => {
      const t = new Tensor([[-1, 2], [-3, 4]]);
      const result = t.relu();
      expect(result.data.get(0, 0)).toBe(0);
      expect(result.data.get(0, 1)).toBe(2);
      expect(result.data.get(1, 0)).toBe(0);
      expect(result.data.get(1, 1)).toBe(4);
    });

    test('relu 反向傳播', () => {
      const t = new Tensor([[-1, 2], [3, 4]], [], true);
      const result = t.relu();
      result.backward();
      expect(t.grad.data[0]).toBe(0);
      expect(t.grad.data[1]).toBe(1);
      expect(t.grad.data[2]).toBe(1);
      expect(t.grad.data[3]).toBe(1);
    });
  });

  describe('masked_fill', () => {
    test('遮罩填充', () => {
      const t = new Tensor([1, 2, 3, 4]);
      const mask = new Tensor([false, true, false, true]);
      const result = t.masked_fill(mask, 0);
      expect(result.data.data[0]).toBe(1);
      expect(result.data.data[1]).toBe(0);
      expect(result.data.data[2]).toBe(3);
      expect(result.data.data[3]).toBe(0);
    });

    test('masked_fill 反向傳播', () => {
      const t = new Tensor([1, 2, 3, 4], [], true);
      const mask = new Tensor([false, true, false, true]);
      const result = t.masked_fill(mask, 0);
      result.backward();
      expect(t.grad.data[0]).toBe(1);
      expect(t.grad.data[1]).toBe(0);
      expect(t.grad.data[2]).toBe(1);
      expect(t.grad.data[3]).toBe(0);
    });
  });

  describe('softmax', () => {
    test('Softmax 機率分布', () => {
      const t = new Tensor([1, 2, 3]);
      const result = t.softmax();
      const sum = result.data.data[0] + result.data.data[1] + result.data.data[2];
      expect(sum).toBeCloseTo(1, 5);
      expect(result.data.data[1]).toBeGreaterThan(result.data.data[0]);
      expect(result.data.data[2]).toBeGreaterThan(result.data.data[1]);
    });

    test('softmax 反向傳播', () => {
      const t = new Tensor([1, 2, 3], [], true);
      const result = t.softmax();
      result.backward();
      for (let i = 0; i < 3; i++) {
        expect(t.grad.data[i]).not.toBeNaN();
      }
    });
  });

  describe('cross_entropy', () => {
    test('交叉熵損失計算', () => {
      const logits = new Tensor([[[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]]]);
      const targets = new Tensor([[1, 1]]);
      const loss = logits.cross_entropy(targets);
      expect(loss.data.data[0]).toBeGreaterThan(0);
    });

    test('cross_entropy 反向傳播', () => {
      const logits = new Tensor([[[2.0, 1.0, 0.1]]], [], true);
      const targets = new Tensor([[1]]);
      const loss = logits.cross_entropy(targets);
      loss.backward();
      expect(logits.grad.data[0]).not.toBeNaN();
    });
  });

  describe('cat', () => {
    test('沿 axis=0 拼接', () => {
      const a = new Tensor([[1, 2], [3, 4]]);
      const b = new Tensor([[5, 6]]);
      const result = cat([a, b], 0);
      expect(result.data.get(0, 0)).toBe(1);
      expect(result.data.get(2, 0)).toBe(5);
    });

    test('cat 反向傳播', () => {
      const a = new Tensor([[1, 2]], [], true);
      const b = new Tensor([[3, 4]], [], true);
      const result = cat([a, b], 0);
      result.backward();
      expect(a.grad.data[0]).toBe(1);
      expect(b.grad.data[0]).toBe(1);
    });
  });
});