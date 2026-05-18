const { train_model, generate_samples } = require('./chargpt.js');
const { GPT } = require('./gpt.js');
const { Adam } = require('./nn.js');

const fs = require('fs');
const path = require('path');

const inputPath = path.join(__dirname, '..', 'input.txt');

let docs;
if (fs.existsSync(inputPath)) {
  const content = fs.readFileSync(inputPath, 'utf-8');
  docs = content.split('\n').filter(line => line.trim());
} else {
  docs = ['alice', 'bob', 'charlie', 'david', 'emma', 'frank', 'grace', 'henry'];
}

console.log(`num docs: ${docs.length}`);

const uchars = [...new Set(docs.join(''))].sort();
const BOS = uchars.length;
const vocab_size = uchars.length + 1;
console.log(`vocab size: ${vocab_size}`);

const block_size = 16;
const model = new GPT(vocab_size, block_size, 1, 16, 4);
console.log(`num params: ${model.parameters().length}`);

const optimizer = new Adam(model.parameters(), lr = 0.01);

console.log('\n--- training ---');
train_model(
  model,
  optimizer,
  docs,
  uchars,
  BOS,
  block_size,
  2000
);

generate_samples(
  model,
  uchars,
  BOS,
  vocab_size,
  block_size,
  5,
  0.5
);

console.log('\nDone!');