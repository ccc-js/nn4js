const { train_model, generate_samples } = require('../src/chargpt.js');
const { GPT } = require('../src/gpt.js');
const { Adam } = require('../src/nn.js');

describe('v0.8: 訓練與推論', () => {
  test('train_model 可執行', () => {
    const docs = ['alice', 'bob', 'charlie'];
    const uchars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'];
    const BOS = uchars.length;
    const vocab_size = uchars.length + 1;

    const model = new GPT(vocab_size, 8, 1, 8, 2);
    const optimizer = new Adam(model.parameters(), 0.01);

    const result = train_model(model, optimizer, docs, uchars, BOS, 8, 2);
    expect(result).toBeDefined();
  });

  test('generate_samples 可執行', () => {
    const docs = ['alice', 'bob'];
    const uchars = ['a', 'b', 'c', 'd', 'e'];
    const BOS = uchars.length;
    const vocab_size = uchars.length + 1;

    const model = new GPT(vocab_size, 8, 1, 8, 2);
    const optimizer = new Adam(model.parameters(), 0.01);

    train_model(model, optimizer, docs, uchars, BOS, 8, 2);
    const results = generate_samples(model, uchars, BOS, vocab_size, 8, 2, 0.5);
    expect(results.length).toBe(2);
  });
});