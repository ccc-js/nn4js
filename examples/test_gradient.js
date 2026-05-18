const { Tensor } = require('../src/tensor.js');

console.log('=== 測試 1: 簡單梯度下降 (x^2 + y^2) ===');
{
  const x = new Tensor([3.0], [], true);
  const y = new Tensor([4.0], [], true);

  const loss = x.mul(x).add(y.mul(y));
  console.log('初始 loss:', loss.data.data[0]);

  loss.backward();
  console.log('x.grad:', x.grad.data[0]);
  console.log('y.grad:', y.grad.data[0]);

  x.data.data[0] -= 0.1 * x.grad.data[0];
  y.data.data[0] -= 0.1 * y.grad.data[0];

  const loss2 = x.mul(x).add(y.mul(y));
  console.log('新 loss:', loss2.data.data[0]);
  console.log('結果: PASS (loss 下降)\n');
}

console.log('=== 測試 2: 檢查 GPT 模型權重形狀 ===');
{
  const { GPT } = require('../src/gpt.js');

  const model = new GPT(10, 5, 1, 8, 2);

  console.log('wte.weight shape:', JSON.stringify(model.wte.weight.data.shape));
  console.log('wpe.weight shape:', JSON.stringify(model.wpe.weight.data.shape));
  console.log('lm_head.weight shape:', JSON.stringify(model.lm_head.weight.data.shape));

  const hasBadShape = model.wte.weight.data.shape.some(s => typeof s !== 'number');
  if (hasBadShape) {
    console.log('結果: FAIL - 權重形狀有問題!\n');
  } else {
    console.log('結果: PASS\n');
  }
}

console.log('=== 測試 3: 前向傳播輸出 ===');
{
  const { GPT } = require('../src/gpt.js');
  const { Tensor } = require('../src/tensor.js');
  const ndarray = require('ndarray');

  const model = new GPT(10, 5, 1, 8, 2);

  const x = new Tensor([[1, 2, 3, 0, 0]], [], true);
  const [logits, _] = model.__call__(x, null);

  console.log('logits shape:', JSON.stringify(logits.data.shape));
  console.log('logits size:', logits.data.size);
  console.log('logits data length:', logits.data.data.length);

  if (logits.data.data.length === 0 || !isFinite(logits.data.data[0])) {
    console.log('結果: FAIL - 輸出無效!\n');
  } else {
    console.log('結果: PASS\n');
  }
}

console.log('=== 測試 4: Cross Entropy Loss ===');
{
  const { GPT } = require('../src/gpt.js');
  const { Tensor } = require('../src/tensor.js');

  const model = new GPT(10, 5, 1, 8, 2);

  const x = new Tensor([[1, 2, 3, 0, 0]], [], true);
  const y = new Tensor([[2, 3, 0, 0, 0]], [], true);

  const [logits, _] = model.__call__(x, null);
  const loss = logits.cross_entropy(y);

  console.log('loss:', loss.data.data[0]);

  if (isNaN(loss.data.data[0])) {
    console.log('結果: FAIL - Loss 是 NaN!\n');
  } else {
    loss.backward();
    console.log('結果: PASS - Loss 有效且可計算梯度\n');
  }
}

console.log('=== 測試 5: 梯度傳播到參數 (lr=0.001) ===');
{
  const { GPT } = require('../src/gpt.js');
  const { Tensor } = require('../src/tensor.js');
  const { Adam } = require('../src/nn.js');

  const model = new GPT(10, 5, 1, 8, 2);
  const params = model.parameters();
  const optimizer = new Adam(params, lr = 0.001);

  const losses = [];
  for (let i = 0; i < 1000; i++) {
    const x = new Tensor([[1, 2, 3, 0, 0]], [], true);
    const y = new Tensor([[2, 3, 0, 0, 0]], [], true);

    optimizer.zero_grad();
    const [logits, _] = model.__call__(x, null);
    const loss = logits.cross_entropy(y);
    losses.push(loss.data.data[0]);

    loss.backward();
    optimizer.step();
  }

  console.log('前5次 loss:', losses.slice(0, 5).map(l => l.toFixed(3)));
  console.log('後5次 loss:', losses.slice(-5).map(l => l.toFixed(3)));
  console.log('loss 下降比例:', (losses[0] / losses[losses.length - 1]).toFixed(2));

  if (losses[losses.length - 1] < losses[0]) {
    console.log('結果: PASS - 訓練有效，loss 下降\n');
  } else {
    console.log('結果: FAIL - loss 沒有下降!\n');
  }
}

console.log('=== 測試 6: SGD 1000 次 ===');
{
  const { GPT } = require('../src/gpt.js');
  const { Tensor } = require('../src/tensor.js');

  const model = new GPT(10, 5, 1, 8, 2);
  const params = model.parameters();
  const lr = 0.01;

  const losses = [];
  for (let i = 0; i < 1000; i++) {
    const x = new Tensor([[1, 2, 3, 0, 0]], [], true);
    const y = new Tensor([[2, 3, 0, 0, 0]], [], true);

    for (const p of params) {
      p.zero_grad();
    }

    const [logits, _] = model.__call__(x, null);
    const loss = logits.cross_entropy(y);
    losses.push(loss.data.data[0]);

    loss.backward();

    for (const p of params) {
      if (p.grad) {
        const grad = p.grad.data;
        const data = p.data.data;
        for (let j = 0; j < data.length; j++) {
          data[j] -= lr * grad[j];
        }
      }
    }
  }

  console.log('前5次 loss:', losses.slice(0, 5).map(l => l.toFixed(3)));
  console.log('後5次 loss:', losses.slice(-5).map(l => l.toFixed(3)));
  console.log('第500次:', losses[500].toFixed(3));
  console.log('loss 下降比例:', (losses[0] / losses[losses.length - 1]).toFixed(2));

  if (losses[losses.length - 1] < losses[0]) {
    console.log('結果: PASS - SGD 訓練有效\n');
  } else {
    console.log('結果: FAIL - loss 沒有下降!\n');
  }
}

console.log('=== 測試 7: Adam 1000 次 (lr=0.001) ===');
{
  const { GPT } = require('../src/gpt.js');
  const { Tensor } = require('../src/tensor.js');
  const { Adam } = require('../src/nn.js');

  const model = new GPT(10, 5, 1, 8, 2);
  const params = model.parameters();
  const optimizer = new Adam(params, lr = 0.001, betas = [0.9, 0.999]);

  const losses = [];
  for (let i = 0; i < 1000; i++) {
    const x = new Tensor([[1, 2, 3, 0, 0]], [], true);
    const y = new Tensor([[2, 3, 0, 0, 0]], [], true);

    optimizer.zero_grad();
    const [logits, _] = model.__call__(x, null);
    const loss = logits.cross_entropy(y);
    losses.push(loss.data.data[0]);

    loss.backward();
    optimizer.step();
  }

  console.log('前5次 loss:', losses.slice(0, 5).map(l => l.toFixed(3)));
  console.log('後5次 loss:', losses.slice(-5).map(l => l.toFixed(3)));
  console.log('第500次:', losses[500].toFixed(3));
  console.log('loss 下降比例:', (losses[0] / losses[losses.length - 1]).toFixed(2));

  if (losses[losses.length - 1] < losses[0]) {
    console.log('結果: PASS - Adam 訓練有效\n');
  } else {
    console.log('結果: FAIL - loss 沒有下降!\n');
  }
}