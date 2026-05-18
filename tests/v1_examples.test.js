const { Tensor } = require('../src/tensor.js');
const { Module, Linear, Adam, setSeed } = require('../src/nn.js');

describe('Linear Regression', () => {
  beforeEach(() => setSeed(42));

  test('single linear layer can learn y = 2x + 1', () => {
    class LinearReg extends Module {
      constructor() {
        super();
        this.linear = new Linear(1, 1, true);
      }
      __call__(x) {
        return this.linear.__call__(x);
      }
    }

    const model = new LinearReg();
    const optimizer = new Adam(model.parameters(), 0.1);

    const trueW = 2, trueB = 1;
    const nSamples = 30;
    const xData = [], yData = [];

    for (let i = 0; i < nSamples; i++) {
      const x = Math.random() * 5;
      const y = trueW * x + trueB + (Math.random() - 0.5) * 2;
      xData.push(x);
      yData.push(y);
    }

    for (let step = 0; step < 50; step++) {
      for (let i = 0; i < nSamples; i++) {
        const x = new Tensor([[xData[i]]], [], true);
        const y = new Tensor([[yData[i]]], [], false);

        optimizer.zero_grad();
        const pred = model.__call__(x);
        const loss = pred.add(y.mul(-1)).mul(pred.add(y.mul(-1)));
        loss.backward();
        optimizer.step();
      }
    }

    const w = model.linear.weight.data.data[0];
    const b = model.linear.bias.data.data[0];

    expect(w).toBeCloseTo(2, 0);
    expect(b).toBeCloseTo(1, 0);
  });
});

describe('XOR Problem', () => {
  beforeEach(() => setSeed(42));

  test('neural network can learn XOR', () => {
    class XORNet extends Module {
      constructor() {
        super();
        this.fc1 = new Linear(2, 8);
        this.fc2 = new Linear(8, 1);
      }
      __call__(x) {
        let out = this.fc1.__call__(x).relu();
        out = this.fc2.__call__(out);
        return out;
      }
    }

    const model = new XORNet();
    const optimizer = new Adam(model.parameters(), 0.1);

    const xorData = [
      [[[0, 0]], [[0]]],
      [[[0, 1]], [[1]]],
      [[[1, 0]], [[1]]],
      [[[1, 1]], [[0]]]
    ];

    for (let step = 0; step < 200; step++) {
      for (const [x_data, y_data] of xorData) {
        const x = new Tensor(x_data, [], true);
        const y = new Tensor(y_data, [], false);

        optimizer.zero_grad();
        const pred = model.__call__(x);
        const loss = pred.add(y.mul(-1)).mul(pred.add(y.mul(-1)));
        loss.backward();
        optimizer.step();
      }
    }

    let correct = 0;
    for (const [x_data, y_data] of xorData) {
      const x = new Tensor(x_data, [], false);
      const pred = model.__call__(x);
      const predVal = pred.data.data[0] > 0.5 ? 1 : 0;
      if (predVal === y_data[0][0]) correct++;
    }

    expect(correct).toBeGreaterThanOrEqual(3);
  });
});

describe('Pattern Classification', () => {
  beforeEach(() => setSeed(123));

  test('network can classify left vs right pattern', () => {
    class PatternNet extends Module {
      constructor() {
        super();
        this.fc1 = new Linear(64, 32);
        this.fc2 = new Linear(32, 2);
      }
      __call__(x) {
        let out = this.fc1.__call__(x).relu();
        out = this.fc2.__call__(out);
        return out;
      }
    }

    const model = new PatternNet();
    const optimizer = new Adam(model.parameters(), 0.1);

    const trainData = [];
    const trainLabels = [];

    for (let i = 0; i < 60; i++) {
      const data = new Array(64).fill(0);
      if (i < 30) {
        for (let j = 0; j < 32; j++) data[j] = 0.9;
        trainLabels.push(0);
      } else {
        for (let j = 32; j < 64; j++) data[j] = 0.9;
        trainLabels.push(1);
      }
      trainData.push(data);
    }

    for (let step = 0; step < 100; step++) {
      for (let i = 0; i < trainData.length; i++) {
        const x = new Tensor([trainData[i]], [], true);
        const y = new Tensor([[trainLabels[i]]], [], false);

        optimizer.zero_grad();
        const logits = model.__call__(x);
        const loss = logits.cross_entropy(y);
        loss.backward();
        optimizer.step();
      }
    }

    let correct = 0;
    for (let i = 0; i < 4; i++) {
      const data = new Array(64).fill(0);
      const expected = i < 2 ? 0 : 1;
      if (i < 2) {
        for (let j = 0; j < 32; j++) data[j] = 0.9;
      } else {
        for (let j = 32; j < 64; j++) data[j] = 0.9;
      }

      const x = new Tensor([data], [], false);
      const logits = model.__call__(x);
      const maxIdx = logits.data.data.indexOf(Math.max(...logits.data.data));

      if (maxIdx === expected) correct++;
    }

    expect(correct).toBeGreaterThanOrEqual(2);
  });
});

