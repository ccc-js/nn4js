const { Tensor } = require('./tensor.js');
const ndarray = require('ndarray');

let seed = 42;
function srandom() {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
}
function setSeed(s) { seed = s; }

function randomNormal(mean = 0, std = 1) {
  let u1 = 0;
  let u2 = 0;
  while (u1 === 0) u1 = srandom();
  while (u2 === 0) u2 = srandom();
  const mag = Math.sqrt(-2.0 * Math.log(u1));
  const z0 = mag * Math.cos(2.0 * Math.PI * u2);
  return mean + std * z0;
}

class Module {
  parameters() {
    const params = [];
    for (const key in this) {
      const v = this[key];
      if (v instanceof Tensor && v.requires_grad) {
        params.push(v);
      } else if (v instanceof Module) {
        params.push(...v.parameters());
      } else if (Array.isArray(v)) {
        for (const item of v) {
          if (item instanceof Module) {
            params.push(...item.parameters());
          }
        }
      }
    }
    return params;
  }
}

class Linear extends Module {
  constructor(in_features, out_features, bias = false) {
    super();
    const std = 0.08;
    const weightData = new Float32Array(in_features * out_features);
    for (let i = 0; i < weightData.length; i++) {
      weightData[i] = randomNormal(0, std);
    }
    this.weight = new Tensor(ndarray(weightData, [in_features, out_features]), [], true);

    if (bias) {
      const biasData = new Float32Array(out_features);
      this.bias = new Tensor(ndarray(biasData, [out_features]), [], true);
    } else {
      this.bias = null;
    }
  }

  __call__(x) {
    let out = x.matmul(this.weight);
    if (this.bias !== null) {
      out = out.add(this.bias);
    }
    return out;
  }
}

class Embedding extends Module {
  constructor(num_embeddings, embedding_dim) {
    super();
    const weightData = new Float32Array(num_embeddings * embedding_dim);
    for (let i = 0; i < weightData.length; i++) {
      weightData[i] = randomNormal(0, 0.08);
    }
    this.weight = new Tensor(ndarray(weightData, [num_embeddings, embedding_dim]), [], true);
  }

  __call__(indices) {
    const indicesT = indices instanceof Tensor ? indices : new Tensor(indices);
    const num_embeddings = this.weight.data.shape[0];
    const embedding_dim = this.weight.data.shape[1];

    const batchSize = indicesT.data.shape[0];
    const seqLen = indicesT.data.shape[1] || 1;

    const outData = new Float32Array(batchSize * seqLen * embedding_dim);
    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen; t++) {
        const idx = indicesT.data.data[b * seqLen + t];
        if (idx >= 0 && idx < num_embeddings) {
          for (let e = 0; e < embedding_dim; e++) {
            outData[(b * seqLen + t) * embedding_dim + e] = this.weight.data.data[idx * embedding_dim + e];
          }
        }
      }
    }

    const out = new Tensor(ndarray(outData, [batchSize, seqLen, embedding_dim]), [this.weight], true);

    const self = this;
    out._backward = () => {
      if (self.weight.requires_grad) {
        const outGradData = out.grad.data;
        for (let b = 0; b < batchSize; b++) {
          for (let t = 0; t < seqLen; t++) {
            const idx = Math.floor(indicesT.data.data[b * seqLen + t]);
            if (idx >= 0 && idx < num_embeddings) {
              for (let e = 0; e < embedding_dim; e++) {
                const outGradIdx = (b * seqLen + t) * embedding_dim + e;
                self.weight.grad.data[idx * embedding_dim + e] += outGradData[outGradIdx];
              }
            }
          }
        }
      }
    };
    return out;
  }
}

class RMSNorm extends Module {
  constructor(dim, eps = 1e-5) {
    super();
    this.eps = eps;
    const scaleData = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      scaleData[i] = 1.0;
    }
    this.scale = new Tensor(ndarray(scaleData, [dim]), [], false);
  }

  __call__(x) {
    const xData = x.data;
    const lastDim = xData.shape[xData.shape.length - 1];
    const outerSize = xData.size / lastDim;

    const msData = new Float32Array(xData.size);
    for (let i = 0; i < outerSize; i++) {
      let sumSq = 0;
      for (let j = 0; j < lastDim; j++) {
        const val = xData.data[i * lastDim + j];
        sumSq += val * val;
      }
      const meanSq = sumSq / lastDim;
      msData[i] = Math.sqrt(meanSq + this.eps);
    }

    const outData = new Float32Array(xData.size);
    for (let i = 0; i < outerSize; i++) {
      const invStd = 1.0 / msData[i];
      for (let j = 0; j < lastDim; j++) {
        outData[i * lastDim + j] = xData.data[i * lastDim + j] * invStd;
      }
    }

    const out = new Tensor(ndarray(outData, xData.shape.slice()), [x], x.requires_grad);

    const self = this;
    out._backward = () => {
      if (x.requires_grad) {
        const xGradData = x.grad.data;
        for (let i = 0; i < outerSize; i++) {
          let sumGradX = 0;
          for (let j = 0; j < lastDim; j++) {
            const idx = i * lastDim + j;
            sumGradX += out.grad.data[idx] * xData.data[idx];
          }

          const invStd = 1.0 / msData[i];
          for (let j = 0; j < lastDim; j++) {
            const idx = i * lastDim + j;
            const term1 = out.grad.data[idx] * invStd;
            const term2 = sumGradX * invStd * invStd * invStd * xData.data[idx] / lastDim;
            xGradData[idx] += term1 - term2;
          }
        }
      }
    };
    return out;
  }
}

class Adam {
  constructor(params, lr = 0.01, betas = [0.85, 0.99], eps = 1e-8) {
    this.params = params;
    this.lr = lr;
    this.beta1 = betas[0];
    this.beta2 = betas[1];
    this.eps = eps;
    this.m = params.map((p) => new Float32Array(p.data.size));
    this.v = params.map((p) => new Float32Array(p.data.size));
    this.t = 0;
  }

  step() {
    this.t += 1;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      const grad = p.grad.data;
      const m = this.m[i];
      const v = this.v[i];

      for (let j = 0; j < p.data.size; j++) {
        m[j] = this.beta1 * m[j] + (1 - this.beta1) * grad[j];
        v[j] = this.beta2 * v[j] + (1 - this.beta2) * grad[j] * grad[j];
        const mHat = m[j] / (1 - Math.pow(this.beta1, this.t));
        const vHat = v[j] / (1 - Math.pow(this.beta2, this.t));
        p.data.data[j] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
      }
    }
  }

  zero_grad() {
    for (const p of this.params) {
      p.zero_grad();
    }
  }
}

module.exports = { Module, Linear, Embedding, RMSNorm, Adam, setSeed };
