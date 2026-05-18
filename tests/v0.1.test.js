const { Tensor } = require('../src/tensor.js');

describe('v0.1: 基礎張量類別', () => {
  test('建構子：建立張量並存取 data', () => {
    const t = new Tensor([1, 2, 3]);
    expect(t.data.size).toBe(3);
    expect(t.data.data).toEqual(new Float32Array([1, 2, 3]));
  });

  test('建構子：支援數值輸入', () => {
    const t = new Tensor(5);
    expect(t.data.size).toBe(1);
    expect(t.data.data[0]).toBe(5);
  });

  test('建構子：支援多維陣列', () => {
    const t = new Tensor([[1, 2], [3, 4]]);
    expect(t.data.size).toBe(4);
    expect(t.data.data).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  test('zero_grad：將梯度歸零', () => {
    const t = new Tensor([1, 2]);
    t.grad.data[0] = 1;
    t.grad.data[1] = 2;
    t.zero_grad();
    expect(t.grad.data).toEqual(new Float32Array([0, 0]));
  });

  test('requires_grad 預設為 false', () => {
    const t = new Tensor([1, 2]);
    expect(t.requires_grad).toBe(false);
  });

  test('requires_grad 可設定為 true', () => {
    const t = new Tensor([1, 2], [], true);
    expect(t.requires_grad).toBe(true);
  });
});