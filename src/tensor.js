const ndarray = require('ndarray');

function computeSize(shape) {
  return shape.length === 0 ? 1 : shape.reduce((a, b) => a * b, 1);
}

function computeStrides(shape) {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

function unravelIndex(index, shape) {
  if (shape.length === 0) return [];
  const strides = computeStrides(shape);
  const indices = new Array(shape.length);
  let remaining = index;
  for (let i = 0; i < shape.length; i++) {
    indices[i] = Math.floor(remaining / strides[i]) % shape[i];
    remaining %= strides[i];
  }
  return indices;
}

function ravelIndex(indices, shape) {
  if (shape.length === 0) return 0;
  const strides = computeStrides(shape);
  let flat = 0;
  for (let i = 0; i < shape.length; i++) {
    flat += indices[i] * strides[i];
  }
  return flat;
}

function broadcastShapes(shapeA, shapeB) {
  const len = Math.max(shapeA.length, shapeB.length);
  const result = new Array(len);
  for (let i = 0; i < len; i++) {
    const a = shapeA[shapeA.length - 1 - i] ?? 1;
    const b = shapeB[shapeB.length - 1 - i] ?? 1;
    if (a !== b && a !== 1 && b !== 1) {
      throw new Error(`Cannot broadcast shapes ${JSON.stringify(shapeA)} and ${JSON.stringify(shapeB)}`);
    }
    result[len - 1 - i] = Math.max(a, b);
  }
  return result;
}

function broadcastIndices(outIndices, srcShape) {
  const offset = outIndices.length - srcShape.length;
  const srcIndices = new Array(srcShape.length);
  for (let i = 0; i < srcShape.length; i++) {
    srcIndices[i] = srcShape[i] === 1 ? 0 : outIndices[offset + i];
  }
  return srcIndices;
}

function getNestedArrayShape(arr) {
  const shape = [];
  let current = arr;
  while (Array.isArray(current)) {
    shape.push(current.length);
    if (current.length > 0) {
      current = current[0];
    } else {
      break;
    }
  }
  return shape.length > 0 ? shape : [arr.length];
}

function broadcastTo(tensor, targetShape) {
  const srcShape = tensor.data.shape.slice();
  const srcData = tensor.data.data;
  const targetSize = computeSize(targetShape);
  const result = new Float32Array(targetSize);

  for (let i = 0; i < targetSize; i++) {
    const outIndices = unravelIndex(i, targetShape);
    const srcIndices = broadcastIndices(outIndices, srcShape);
    result[i] = srcData[ravelIndex(srcIndices, srcShape)];
  }
  return ndarray(result, targetShape);
}

function getIndex(arr, indices) {
  let idx = 0;
  let multiplier = 1;
  const ndim = arr.shape.length;
  for (let i = ndim - 1; i >= 0; i--) {
    const idxVal = indices[i] !== undefined ? indices[i] : 0;
    idx += idxVal * multiplier;
    multiplier *= arr.shape[i];
  }
  return arr.data[idx];
}

function unbroadcast(grad, shape) {
  const gradShape = grad.shape.slice();
  if (JSON.stringify(gradShape) === JSON.stringify(shape)) {
    return grad;
  }

  const outData = new Float32Array(computeSize(shape));
  const paddedShape = Array(Math.max(0, gradShape.length - shape.length)).fill(1).concat(shape);

  for (let i = 0; i < grad.data.length; i++) {
    const gradIndices = unravelIndex(i, gradShape);
    const targetPaddedIndices = gradIndices.map((idx, dim) => (paddedShape[dim] === 1 ? 0 : idx));
    const targetIndices = targetPaddedIndices.slice(targetPaddedIndices.length - shape.length);
    outData[ravelIndex(targetIndices, shape)] += grad.data[i];
  }

  return ndarray(outData, shape.slice());
}

class Tensor {
  constructor(data, _children = [], requires_grad = false) {
    if (data && data.shape && data.data) {
      this.data = data;
    } else if (data instanceof Float32Array) {
      this.data = ndarray(data);
    } else if (typeof data === 'number') {
      this.data = ndarray(new Float32Array([data]));
    } else if (Array.isArray(data)) {
      const flat = data.flat(Infinity);
      const shape = getNestedArrayShape(data);
      this.data = ndarray(new Float32Array(flat), shape);
    } else {
      throw new Error('Invalid data type for Tensor');
    }
    this.grad = ndarray(new Float32Array(this.data.size), this.data.shape.slice());
    this._backward = () => {};
    this._prev = new Set(_children);
    this.requires_grad = requires_grad;
  }

  zero_grad() {
    const size = this.grad.size;
    this.grad = ndarray(new Float32Array(size), this.data.shape.slice());
  }

  backward() {
    const topo = [];
    const visited = new Set();
    const build_topo = (v) => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._prev) {
          build_topo(child);
        }
        topo.push(v);
      }
    };
    build_topo(this);

    const oneArr = new Float32Array(this.data.size).fill(1);
    this.grad = ndarray(oneArr, this.data.shape.slice());

    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }

  add(other) {
    const otherT = other instanceof Tensor ? other : new Tensor(other);
    const outShape = broadcastShapes(this.data.shape, otherT.data.shape);
    const selfBroadcast = broadcastTo(this, outShape);
    const otherBroadcast = broadcastTo(otherT, outShape);
    const outData = new Float32Array(computeSize(outShape));
    for (let i = 0; i < outData.length; i++) {
      outData[i] = selfBroadcast.data[i] + otherBroadcast.data[i];
    }
    const out = new Tensor(ndarray(outData, outShape.slice()), [this, otherT], this.requires_grad || otherT.requires_grad);
    const self = this;
    const otherTensor = otherT;
    out._backward = () => {
      if (self.requires_grad) {
        const g = unbroadcast(out.grad, self.data.shape);
        for (let i = 0; i < self.grad.size; i++) {
          self.grad.data[i] += g.data[i];
        }
      }
      if (otherTensor.requires_grad) {
        const g = unbroadcast(out.grad, otherTensor.data.shape);
        for (let i = 0; i < otherTensor.grad.size; i++) {
          otherTensor.grad.data[i] += g.data[i];
        }
      }
    };
    return out;
  }

  mul(other) {
    const otherT = other instanceof Tensor ? other : new Tensor(other);
    const outShape = broadcastShapes(this.data.shape, otherT.data.shape);
    const selfBroadcast = broadcastTo(this, outShape);
    const otherBroadcast = broadcastTo(otherT, outShape);
    const outData = new Float32Array(computeSize(outShape));
    for (let i = 0; i < outData.length; i++) {
      outData[i] = selfBroadcast.data[i] * otherBroadcast.data[i];
    }
    const out = new Tensor(ndarray(outData, outShape.slice()), [this, otherT], this.requires_grad || otherT.requires_grad);
    const self = this;
    const otherTensor = otherT;
    out._backward = () => {
      if (self.requires_grad) {
        const selfGradData = new Float32Array(outData.length);
        for (let i = 0; i < outData.length; i++) {
          selfGradData[i] = out.grad.data[i] * otherBroadcast.data[i];
        }
        const g = unbroadcast(ndarray(selfGradData, outShape.slice()), self.data.shape);
        for (let i = 0; i < self.grad.size; i++) {
          self.grad.data[i] += g.data[i];
        }
      }
      if (otherTensor.requires_grad) {
        const otherGradData = new Float32Array(outData.length);
        for (let i = 0; i < outData.length; i++) {
          otherGradData[i] = out.grad.data[i] * selfBroadcast.data[i];
        }
        const g = unbroadcast(ndarray(otherGradData, outShape.slice()), otherTensor.data.shape);
        for (let i = 0; i < otherTensor.grad.size; i++) {
          otherTensor.grad.data[i] += g.data[i];
        }
      }
    };
    return out;
  }

  matmul(other) {
    const otherT = other instanceof Tensor ? other : new Tensor(other);
    const a = this.data;
    const b = otherT.data;
    const aShape = a.shape.slice();
    const bShape = b.shape.slice();
    if (aShape.length < 2 || bShape.length < 2) {
      throw new Error('matmul requires tensors with at least 2 dimensions');
    }

    const M = aShape[aShape.length - 2];
    const K = aShape[aShape.length - 1];
    const K2 = bShape[bShape.length - 2];
    const N = bShape[bShape.length - 1];
    if (K !== K2) {
      throw new Error(`matmul shape mismatch: ${JSON.stringify(aShape)} x ${JSON.stringify(bShape)}`);
    }

    const batchShape = broadcastShapes(aShape.slice(0, -2), bShape.slice(0, -2));
    const outShape = batchShape.concat([M, N]);
    const outData = new Float32Array(computeSize(outShape));
    const batchSize = computeSize(batchShape);
    const self = this;
    const otherTensor = otherT;

    for (let batch = 0; batch < batchSize; batch++) {
      const batchIndices = unravelIndex(batch, batchShape);
      const aBatch = broadcastIndices(batchIndices, aShape.slice(0, -2));
      const bBatch = broadcastIndices(batchIndices, bShape.slice(0, -2));
      for (let m = 0; m < M; m++) {
        for (let n = 0; n < N; n++) {
          let sum = 0;
          for (let k = 0; k < K; k++) {
            sum += getIndex(a, aBatch.concat([m, k])) * getIndex(b, bBatch.concat([k, n]));
          }
          outData[ravelIndex(batchIndices.concat([m, n]), outShape)] = sum;
        }
      }
    }

    const out = new Tensor(ndarray(outData, outShape), [this, otherT], this.requires_grad || otherT.requires_grad);
    out._backward = () => {
      if (self.requires_grad) {
        const gradA = new Float32Array(a.size);
        for (let batch = 0; batch < batchSize; batch++) {
          const batchIndices = unravelIndex(batch, batchShape);
          const aBatch = broadcastIndices(batchIndices, aShape.slice(0, -2));
          const bBatch = broadcastIndices(batchIndices, bShape.slice(0, -2));
          for (let m = 0; m < M; m++) {
            for (let k = 0; k < K; k++) {
              let sum = 0;
              for (let n = 0; n < N; n++) {
                sum += getIndex(out.grad, batchIndices.concat([m, n])) * getIndex(b, bBatch.concat([k, n]));
              }
              gradA[ravelIndex(aBatch.concat([m, k]), aShape)] += sum;
            }
          }
        }
        for (let i = 0; i < self.grad.size; i++) {
          self.grad.data[i] += gradA[i];
        }
      }
      if (otherTensor.requires_grad) {
        const gradB = new Float32Array(b.size);
        for (let batch = 0; batch < batchSize; batch++) {
          const batchIndices = unravelIndex(batch, batchShape);
          const aBatch = broadcastIndices(batchIndices, aShape.slice(0, -2));
          const bBatch = broadcastIndices(batchIndices, bShape.slice(0, -2));
          for (let k = 0; k < K; k++) {
            for (let n = 0; n < N; n++) {
              let sum = 0;
              for (let m = 0; m < M; m++) {
                sum += getIndex(a, aBatch.concat([m, k])) * getIndex(out.grad, batchIndices.concat([m, n]));
              }
              gradB[ravelIndex(bBatch.concat([k, n]), bShape)] += sum;
            }
          }
        }
        for (let i = 0; i < otherTensor.grad.size; i++) {
          otherTensor.grad.data[i] += gradB[i];
        }
      }
    };
    return out;
  }

  transpose(...axes) {
    const shape = this.data.shape.slice();
    let perm;
    if (axes.length === 2) {
      const idx1 = axes[0] < 0 ? shape.length + axes[0] : axes[0];
      const idx2 = axes[1] < 0 ? shape.length + axes[1] : axes[1];
      perm = shape.map((_, i) => i);
      [perm[idx1], perm[idx2]] = [perm[idx2], perm[idx1]];
    } else if (axes.length === shape.length) {
      perm = axes.map((ax) => (ax < 0 ? shape.length + ax : ax));
    } else {
      throw new Error(`transpose expected 2 axes or full permutation, got ${axes.length}`);
    }

    const newShape = perm.map((idx) => shape[idx]);
    const outSize = computeSize(newShape);
    const transData = new Float32Array(outSize);

    for (let i = 0; i < outSize; i++) {
      const outIndices = unravelIndex(i, newShape);
      const srcIndices = new Array(shape.length);
      for (let j = 0; j < perm.length; j++) {
        srcIndices[perm[j]] = outIndices[j];
      }
      transData[i] = this.data.data[ravelIndex(srcIndices, shape)];
    }

    const out = new Tensor(ndarray(transData, newShape), [this], this.requires_grad);
    const self = this;
    const inversePerm = new Array(perm.length);
    for (let i = 0; i < perm.length; i++) {
      inversePerm[perm[i]] = i;
    }
    out._backward = () => {
      if (self.requires_grad) {
        const gradTrans = new Float32Array(self.data.size);
        for (let i = 0; i < out.grad.data.length; i++) {
          const outIndices = unravelIndex(i, newShape);
          const srcIndices = inversePerm.map((pos) => outIndices[pos]);
          gradTrans[ravelIndex(srcIndices, shape)] += out.grad.data[i];
        }
        for (let i = 0; i < self.grad.size; i++) {
          self.grad.data[i] += gradTrans[i];
        }
      }
    };
    return out;
  }

  reshape(...shape) {
    const newSize = shape.reduce((a, b) => a * b, 1);
    const outData = new Float32Array(newSize);
    for (let i = 0; i < newSize && i < this.data.size; i++) {
      outData[i] = this.data.data[i];
    }
    const out = new Tensor(ndarray(outData, shape), [this], this.requires_grad);
    const self = this;
    out._backward = () => {
      if (self.requires_grad) {
        const gradReshaped = out.grad.data.slice(0, self.data.size);
        for (let i = 0; i < self.grad.size; i++) {
          self.grad.data[i] += gradReshaped[i];
        }
      }
    };
    return out;
  }

  relu() {
    const outData = new Float32Array(this.data.size);
    for (let i = 0; i < this.data.size; i++) {
      outData[i] = Math.max(0, this.data.data[i]);
    }
    const out = new Tensor(ndarray(outData, this.data.shape.slice()), [this], this.requires_grad);
    const self = this;
    out._backward = () => {
      if (self.requires_grad) {
        for (let i = 0; i < self.data.size; i++) {
          if (self.data.data[i] > 0) {
            self.grad.data[i] += out.grad.data[i];
          }
        }
      }
    };
    return out;
  }

  masked_fill(mask, value) {
    const maskT = mask instanceof Tensor ? mask : new Tensor(mask);
    const maskBroadcast = broadcastTo(maskT, this.data.shape.slice());
    const outData = new Float32Array(this.data.size);
    for (let i = 0; i < this.data.size; i++) {
      outData[i] = maskBroadcast.data[i] ? value : this.data.data[i];
    }
    const out = new Tensor(ndarray(outData, this.data.shape.slice()), [this], this.requires_grad);
    const self = this;
    out._backward = () => {
      if (self.requires_grad) {
        for (let i = 0; i < self.data.size; i++) {
          if (!maskBroadcast.data[i]) {
            self.grad.data[i] += out.grad.data[i];
          }
        }
      }
    };
    return out;
  }

  softmax(axis = -1) {
    const shape = this.data.shape;
    const axisPos = axis < 0 ? shape.length + axis : axis;
    const innerSize = shape[axisPos];
    const trailingSize = computeSize(shape.slice(axisPos + 1));
    const outerSize = computeSize(shape.slice(0, axisPos));
    const blockCount = outerSize * trailingSize;

    const maxData = new Float32Array(this.data.size);
    for (let block = 0; block < blockCount; block++) {
      const outer = Math.floor(block / trailingSize);
      const trailing = block % trailingSize;
      let maxVal = -Infinity;
      for (let j = 0; j < innerSize; j++) {
        const idx = outer * innerSize * trailingSize + j * trailingSize + trailing;
        if (this.data.data[idx] > maxVal) maxVal = this.data.data[idx];
      }
      for (let j = 0; j < innerSize; j++) {
        const idx = outer * innerSize * trailingSize + j * trailingSize + trailing;
        maxData[idx] = maxVal;
      }
    }

    const expData = new Float32Array(this.data.size);
    const outData = new Float32Array(this.data.size);
    for (let block = 0; block < blockCount; block++) {
      const outer = Math.floor(block / trailingSize);
      const trailing = block % trailingSize;
      let sumExp = 0;
      for (let j = 0; j < innerSize; j++) {
        const idx = outer * innerSize * trailingSize + j * trailingSize + trailing;
        expData[idx] = Math.exp(this.data.data[idx] - maxData[idx]);
        sumExp += expData[idx];
      }
      for (let j = 0; j < innerSize; j++) {
        const idx = outer * innerSize * trailingSize + j * trailingSize + trailing;
        outData[idx] = expData[idx] / sumExp;
      }
    }

    const out = new Tensor(ndarray(outData, shape.slice()), [this], this.requires_grad);
    const self = this;
    out._backward = () => {
      if (self.requires_grad) {
        for (let block = 0; block < blockCount; block++) {
          const outer = Math.floor(block / trailingSize);
          const trailing = block % trailingSize;
          let gradSum = 0;
          for (let k = 0; k < innerSize; k++) {
            const idx = outer * innerSize * trailingSize + k * trailingSize + trailing;
            gradSum += out.grad.data[idx] * out.data.data[idx];
          }
          for (let j = 0; j < innerSize; j++) {
            const idx = outer * innerSize * trailingSize + j * trailingSize + trailing;
            const softmaxVal = out.data.data[idx];
            self.grad.data[idx] += out.grad.data[idx] * softmaxVal - gradSum * softmaxVal;
          }
        }
      }
    };
    return out;
  }

  cross_entropy(targets) {
    const targetsT = targets instanceof Tensor ? targets : new Tensor(targets);
    const logits = this.data;
    const batchSize = targetsT.data.shape[0];
    const seqLen = targetsT.data.shape[1];
    const vocabSize = logits.shape[2];

    let loss = 0;
    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen; t++) {
        const targetIdx = targetsT.data.data[b * seqLen + t];
        let maxLogit = -Infinity;
        for (let v = 0; v < vocabSize; v++) {
          if (logits.data[b * seqLen * vocabSize + t * vocabSize + v] > maxLogit) {
            maxLogit = logits.data[b * seqLen * vocabSize + t * vocabSize + v];
          }
        }
        let sumExp = 0;
        const exps = new Float32Array(vocabSize);
        for (let v = 0; v < vocabSize; v++) {
          exps[v] = Math.exp(logits.data[b * seqLen * vocabSize + t * vocabSize + v] - maxLogit);
          sumExp += exps[v];
        }
        const prob = exps[targetIdx] / sumExp;
        loss -= Math.log(prob + 1e-10);
      }
    }
    loss = loss / (batchSize * seqLen);

    const out = new Tensor(loss, [this], this.requires_grad);
    const self = this;
    out._backward = () => {
      if (self.requires_grad) {
        for (let b = 0; b < batchSize; b++) {
          for (let t = 0; t < seqLen; t++) {
            const targetIdx = targetsT.data.data[b * seqLen + t];
            let maxLogit = -Infinity;
            for (let v = 0; v < vocabSize; v++) {
              const idx = b * seqLen * vocabSize + t * vocabSize + v;
              if (logits.data[idx] > maxLogit) maxLogit = logits.data[idx];
            }
            let sumExp = 0;
            const probs = new Float32Array(vocabSize);
            for (let v = 0; v < vocabSize; v++) {
              const idx = b * seqLen * vocabSize + t * vocabSize + v;
              probs[v] = Math.exp(logits.data[idx] - maxLogit);
              sumExp += probs[v];
            }
            for (let v = 0; v < vocabSize; v++) {
              probs[v] = probs[v] / sumExp;
            }
            probs[targetIdx] -= 1;
            const gradScale = 1 / (batchSize * seqLen);
            for (let v = 0; v < vocabSize; v++) {
              const idx = b * seqLen * vocabSize + t * vocabSize + v;
              self.grad.data[idx] += out.grad.data[0] * probs[v] * gradScale;
            }
          }
        }
      }
    };
    return out;
  }
}

function cat(tensors, axis = 0) {
  const shapes = tensors.map((t) => t.data.shape);
  const outShape = shapes[0].slice();
  for (let i = 1; i < shapes.length; i++) {
    for (let d = 0; d < outShape.length; d++) {
      if (d !== axis && shapes[i][d] !== outShape[d]) {
        throw new Error(`cat shape mismatch on axis ${axis}: ${JSON.stringify(shapes)}`);
      }
    }
  }

  const axisSize = shapes.reduce((sum, s) => sum + s[axis], 0);
  outShape[axis] = axisSize;

  const outData = new Float32Array(computeSize(outShape));
  let axisOffset = 0;
  for (const tensor of tensors) {
    for (let i = 0; i < tensor.data.size; i++) {
      const srcIndices = unravelIndex(i, tensor.data.shape);
      const outIndices = srcIndices.slice();
      outIndices[axis] += axisOffset;
      outData[ravelIndex(outIndices, outShape)] = tensor.data.data[i];
    }
    axisOffset += tensor.data.shape[axis];
  }

  const requiresGrad = tensors.some((t) => t.requires_grad);
  const out = new Tensor(ndarray(outData, outShape), tensors, requiresGrad);

  out._backward = () => {
    if (!out.requires_grad) return;
    let gradAxisOffset = 0;
    for (const t of tensors) {
      if (t.requires_grad) {
        for (let j = 0; j < t.data.size; j++) {
          const srcIndices = unravelIndex(j, t.data.shape);
          const outIndices = srcIndices.slice();
          outIndices[axis] += gradAxisOffset;
          t.grad.data[j] += out.grad.data[ravelIndex(outIndices, outShape)];
        }
      }
      gradAxisOffset += t.data.shape[axis];
    }
  };
  return out;
}

module.exports = { Tensor, cat };
