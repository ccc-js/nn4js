const { train_model, generate_samples } = require('../src/chargpt.js');
const { GPT } = require('../src/gpt.js');
const { Adam, setSeed } = require('../src/nn.js');

describe('CharGPT Chinese Training', () => {
  beforeEach(() => setSeed(42));

  test('can train on Chinese sentences', () => {
    const docs = [
      '小貓', '小狗', '小鳥', '小魚',
      '大貓', '大狗', '大鳥', '大魚'
    ];

    const uchars = [...new Set(docs.join(''))].sort();
    const BOS = uchars.length;
    const vocab_size = uchars.length + 1;
    const block_size = 16;

    const model = new GPT(vocab_size, block_size, 1, 16, 4);
    const optimizer = new Adam(model.parameters(), 0.01);

    train_model(model, optimizer, docs, uchars, BOS, block_size, 50);

    expect(model.parameters().length).toBeGreaterThan(0);
  });

  test('can generate samples after training', () => {
    const docs = ['張三', '李四', '王五', '陳六'];
    const uchars = [...new Set(docs.join(''))].sort();
    const BOS = uchars.length;
    const vocab_size = uchars.length + 1;
    const block_size = 8;

    const model = new GPT(vocab_size, block_size, 1, 16, 2);
    const optimizer = new Adam(model.parameters(), 0.01);

    train_model(model, optimizer, docs, uchars, BOS, block_size, 30);
    const results = generate_samples(model, uchars, BOS, vocab_size, block_size, 3, 0.5);

    expect(results.length).toBe(3);
    results.forEach(name => {
      expect(typeof name).toBe('string');
      expect(name.length).toBeGreaterThan(0);
    });
  });
});