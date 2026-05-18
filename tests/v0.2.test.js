const { Tensor } = require('../src/tensor.js');

describe('v0.2: 張量運算與自動微分', () => {
  test('add：兩個張量相加', () => {
    const a = new Tensor([1, 2, 3], [], true);
    const b = new Tensor([4, 5, 6], [], true);
    const c = a.add(b);
    expect(c.data.data).toEqual(new Float32Array([5, 7, 9]));
  });

  test('add：張量加純量', () => {
    const a = new Tensor([1, 2, 3], [], true);
    const c = a.add(1);
    expect(c.data.data).toEqual(new Float32Array([2, 3, 4]));
  });

  test('mul：兩個張量相乘', () => {
    const a = new Tensor([1, 2, 3], [], true);
    const b = new Tensor([4, 5, 6], [], true);
    const c = a.mul(b);
    expect(c.data.data).toEqual(new Float32Array([4, 10, 18]));
  });

  test('mul：張量乘純量', () => {
    const a = new Tensor([1, 2, 3], [], true);
    const c = a.mul(2);
    expect(c.data.data).toEqual(new Float32Array([2, 4, 6]));
  });

  test('matmul：矩陣乘法', () => {
    const a = new Tensor([[1, 2], [3, 4]], [], true);
    const b = new Tensor([[5, 6], [7, 8]], [], true);
    const c = a.matmul(b);
    expect(c.data.get(0, 0)).toBe(19);
    expect(c.data.get(0, 1)).toBe(22);
    expect(c.data.get(1, 0)).toBe(43);
    expect(c.data.get(1, 1)).toBe(50);
  });

  test('backward：反向傳播梯度', () => {
    const a = new Tensor([1, 2], [], true);
    const b = new Tensor([3, 4], [], true);
    const c = a.add(b);
    c.backward();
    expect(a.grad.data).toEqual(new Float32Array([1, 1]));
    expect(b.grad.data).toEqual(new Float32Array([1, 1]));
  });

  test('backward：mul 反向傳播', () => {
    const a = new Tensor([2, 3], [], true);
    const b = new Tensor([4, 5], [], true);
    const c = a.mul(b);
    c.backward();
    expect(a.grad.data).toEqual(new Float32Array([4, 5]));
    expect(b.grad.data).toEqual(new Float32Array([2, 3]));
  });

  test('backward：matmul 反向傳播', () => {
    const a = new Tensor([[1, 2], [3, 4]], [], true);
    const b = new Tensor([[5], [6]], [], true);
    const c = a.matmul(b);
    c.backward();
    expect(a.grad.data[0]).toBe(5);
    expect(a.grad.data[1]).toBe(6);
  });

  test('chained operations：鏈式運算', () => {
    const a = new Tensor([1, 2], [], true);
    const b = new Tensor([3, 4], [], true);
    const c = a.add(b).mul(a);
    c.backward();
    expect(a.grad.data[0]).toBe(5);
    expect(a.grad.data[1]).toBe(8);
  });
});