import * as ort from 'onnxruntime-web';

ort.env.wasm.proxy = false;
ort.env.wasm.numThreads = 1;

export type OnnxConfig = {
  path: string;
  meta?: {
    in_keys?: string[];
    out_keys?: (string | string[])[];
  };
};

export class OnnxModule {
  private config: OnnxConfig;
  private session: ort.InferenceSession | null;
  private configuredInKeys: string[];
  inKeys: string[];
  outKeys: string[];
  isRecurrent: boolean;

  constructor(config: OnnxConfig) {
    if (!config?.path) {
      throw new Error('OnnxModule requires a path.');
    }
    this.config = config;
    this.session = null;
    const inKeys = config.meta?.in_keys ?? ['policy'];
    const outKeys = config.meta?.out_keys ?? ['action'];
    this.configuredInKeys = inKeys.map((key) => (Array.isArray(key) ? key.join(',') : key));
    this.inKeys = [...this.configuredInKeys];
    this.outKeys = outKeys.map((key) => (Array.isArray(key) ? key.join(',') : key));
    this.isRecurrent = this.inKeys.includes('adapt_hx');
  }

  async init(): Promise<void> {
    const response = await fetch(this.config.path);
    if (!response.ok) {
      throw new Error(`Failed to fetch ONNX model: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    this.session = await ort.InferenceSession.create(buffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    this.inferInputKeys();
    this.isRecurrent = this.inKeys.includes('adapt_hx');
  }

  initInput(): Record<string, ort.Tensor> {
    const inputs: Record<string, ort.Tensor> = {};
    if (this.isRecurrent) {
      inputs.is_init = new ort.Tensor('bool', [true], [1]);
      inputs.adapt_hx = new ort.Tensor('float32', new Float32Array(128), [1, 128]);
    }
    if (this.inKeys.includes('time_step')) {
      inputs.time_step = new ort.Tensor('float32', new Float32Array([0.0]), [1, 1]);
    }
    return inputs;
  }

  async runInference(
    input: Record<string, ort.Tensor>
  ): Promise<[Record<string, ort.Tensor>, Record<string, ort.Tensor>]> {
    if (!this.session) {
      throw new Error('OnnxModule not initialized.');
    }

    const onnxInput: Record<string, ort.Tensor> = {};
    for (let i = 0; i < this.inKeys.length; i++) {
      const key = this.inKeys[i];
      const name = this.session.inputNames[i];
      if (!name || !input[key]) {
        throw new Error(`Missing ONNX input for key: ${key}`);
      }
      onnxInput[name] = input[key];
    }

    const onnxOutput = await this.session.run(onnxInput);
    const result: Record<string, ort.Tensor> = {};
    for (let i = 0; i < this.outKeys.length; i++) {
      const key = this.outKeys[i];
      const name = this.session.outputNames[i];
      if (name && onnxOutput[name]) {
        result[key] = onnxOutput[name];
      }
    }

    const carry: Record<string, ort.Tensor> = {};
    if (this.isRecurrent && result['next,adapt_hx']) {
      carry.is_init = new ort.Tensor('bool', [false], [1]);
      carry.adapt_hx = result['next,adapt_hx'];
    }

    return [result, carry];
  }

  private inferInputKeys(): void {
    if (!this.session) {
      return;
    }
    const modelInputs = this.session.inputNames;
    if (modelInputs.length <= 1) {
      this.inKeys = [...this.configuredInKeys];
      return;
    }

    if (this.configuredInKeys.length === modelInputs.length) {
      this.inKeys = [...this.configuredInKeys];
      return;
    }

    this.inKeys = modelInputs.map((name) => {
      if (this.configuredInKeys.includes(name)) {
        return name;
      }
      if (name === 'obs' && this.configuredInKeys.includes('policy')) {
        return 'policy';
      }
      // No configured key matches this model input by name; pass it through as-is.
      // runInference will forward it using the model's own input name.
      return name;
    });
  }
}
