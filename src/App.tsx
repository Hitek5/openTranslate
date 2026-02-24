import { useEffect, useMemo, useState } from 'react';
import { pipeline } from '@xenova/transformers';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';

type Timestamp = [number, number];

type Chunk = {
  text: string;
  timestamp: Timestamp;
};

type TranscriptResult = {
  text: string;
  chunks?: Chunk[];
};

type Status = 'idle' | 'loading-model' | 'transcribing' | 'done' | 'error';
type Device = 'wasm' | 'webgpu';

type ProgressInfo = {
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
  status?: string;
};

const MODELS = [
  { value: 'Xenova/whisper-tiny', label: 'Whisper Tiny (быстро)' },
  { value: 'Xenova/whisper-base', label: 'Whisper Base (баланс)' },
  { value: 'onnx-community/whisper-large-v3-turbo', label: 'Whisper Large v3 Turbo (точнее, тяжелее)' },
] as const;

const DEVICES: { value: Device; label: string; hint: string }[] = [
  { value: 'wasm', label: 'CPU (WASM)', hint: 'Стабильно на любом ПК, но медленнее.' },
  { value: 'webgpu', label: 'GPU (WebGPU)', hint: 'Быстрее, если браузер и видеокарта поддерживают WebGPU.' },
];

const STALL_WARNING_MS = 20_000;

let currentCacheKey = '';
let currentAsr: Awaited<ReturnType<typeof pipeline>> | null = null;

const formatSeconds = (value: number): string => {
  const safeValue = Math.max(0, Number.isFinite(value) ? value : 0);
  const hrs = Math.floor(safeValue / 3600);
  const min = Math.floor((safeValue % 3600) / 60);
  const sec = Math.floor(safeValue % 60);
  const ms = Math.floor((safeValue % 1) * 1000);
  return `${String(hrs).padStart(2, '0')}:${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
};

const toSrt = (chunks: Chunk[]): string =>
  chunks
    .map((chunk, index) => {
      const [start, end] = chunk.timestamp;
      return `${index + 1}\n${formatSeconds(start)} --> ${formatSeconds(end)}\n${chunk.text.trim()}\n`;
    })
    .join('\n');

const getNormalizedPercent = (value?: number): number | null => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return null;
  }

  if (value <= 1) {
    return Math.min(100, Math.max(0, Math.round(value * 100)));
  }

  return Math.min(100, Math.max(0, Math.round(value)));
};

const decodeFileToMonoFloat32 = async (file: File): Promise<Float32Array> => {
  const arrayBuffer = await file.arrayBuffer();
  const audioContext = new AudioContext({ sampleRate: 16_000 });
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const channels = audioBuffer.numberOfChannels;
    const length = audioBuffer.length;

    if (channels === 1) {
      return new Float32Array(audioBuffer.getChannelData(0));
    }

    const mono = new Float32Array(length);
    for (let channel = 0; channel < channels; channel += 1) {
      const data = audioBuffer.getChannelData(channel);
      for (let i = 0; i < length; i += 1) {
        mono[i] += data[i] / channels;
      }
    }
    return mono;
  } finally {
    await audioContext.close();
  }
};

const App = () => {
  const [selectedModel, setSelectedModel] = useState<(typeof MODELS)[number]['value']>(MODELS[0].value);
  const [selectedDevice, setSelectedDevice] = useState<Device>('wasm');
  const [status, setStatus] = useState<Status>('idle');
  const [statusText, setStatusText] = useState('Выберите файл и нажмите «Транскрибировать».');
  const [statusHint, setStatusHint] = useState('');
  const [statusPercent, setStatusPercent] = useState<number | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [transcript, setTranscript] = useState<TranscriptResult | null>(null);
  const [lastProgressAt, setLastProgressAt] = useState<number>(0);

  const canTranscribe = useMemo(() => !!file && (status === 'idle' || status === 'done' || status === 'error'), [file, status]);

  useEffect(() => {
    if (status !== 'loading-model') {
      return;
    }

    const timer = window.setInterval(() => {
      const stalled = Date.now() - lastProgressAt > STALL_WARNING_MS;
      if (stalled) {
        setStatusHint('Загрузка долго стоит на одном файле. Обычно помогает: подождать 1-2 минуты, переключить GPU→CPU (WASM), отключить VPN/прокси и попробовать модель Tiny/Base.');
      }
    }, 1000);

    return () => window.clearInterval(timer);
  }, [lastProgressAt, status]);

  const loadPipeline = async (device: Device) => {
    const cacheKey = `${selectedModel}::${device}`;
    if (currentAsr && currentCacheKey === cacheKey) return currentAsr;

    setStatus('loading-model');
    setStatusText(`Загружаем модель (${device === 'webgpu' ? 'GPU' : 'CPU'})...`);
    setStatusHint('');
    setStatusPercent(0);
    setLastProgressAt(Date.now());

    const fileProgress = new Map<string, number>();

    currentAsr = await pipeline('automatic-speech-recognition', selectedModel, ({
      device,
      progress_callback: (progress: ProgressInfo) => {
        const part = progress.file ?? progress.status ?? 'модель';
        setLastProgressAt(Date.now());

        let currentPercent = getNormalizedPercent(progress.progress);
        if (currentPercent === null && typeof progress.loaded === 'number' && typeof progress.total === 'number' && progress.total > 0) {
          currentPercent = getNormalizedPercent((progress.loaded / progress.total) * 100);
        }

        if (currentPercent !== null) {
          fileProgress.set(part, currentPercent);
          const values = [...fileProgress.values()];
          const average = Math.round(values.reduce((sum, item) => sum + item, 0) / values.length);
          setStatusPercent(average);
          setStatusText(`Загрузка: ${part} (${currentPercent}%)`);
        } else {
          setStatusText(`Загрузка: ${part}`);
        }
      },
    }) as any);

    currentCacheKey = cacheKey;
    setStatusPercent(100);
    return currentAsr;
  };

  const runTranscription = async (deviceToUse: Device) => {
    if (!file) return;

    const asr = await loadPipeline(deviceToUse);

    setStatus('transcribing');
    setStatusText('Модель загружена. Подготавливаем аудио...');
    setStatusHint('');
    setStatusPercent(null);

    const audioData = await decodeFileToMonoFloat32(file);

    setStatusText('Расшифровываем аудио...');

    const result = (await (asr as any)(audioData, {
      chunk_length_s: 30,
      stride_length_s: 5,
      return_timestamps: true,
      language: 'russian',
      task: 'transcribe',
    })) as TranscriptResult;

    setTranscript(result);
    setStatus('done');
    setStatusPercent(100);
    setStatusText('Готово! Можно скачать TXT/SRT или скопировать текст.');
    setStatusHint('');
  };

  const handleTranscribe = async () => {
    if (!file) return;

    setTranscript(null);
    setStatus('transcribing');
    setStatusText('Идёт распознавание речи...');
    setStatusHint('');
    setStatusPercent(null);

    try {
      await runTranscription(selectedDevice);
    } catch (error) {
      console.error(error);

      if (selectedDevice === 'webgpu') {
        try {
          setStatus('loading-model');
          setStatusText('GPU недоступен или нестабилен. Переключаемся на CPU (WASM)...');
          setStatusHint('Авто-фолбэк активирован: повторяем загрузку/транскрибацию на CPU.');
          await runTranscription('wasm');
          setSelectedDevice('wasm');
          return;
        } catch (fallbackError) {
          console.error(fallbackError);
        }
      }

      setStatus('error');
      setStatusPercent(null);
      setStatusText('Ошибка транскрибации. Если загрузка зависает на config.json/tokenizer_config.json — попробуйте CPU (WASM), Tiny/Base и проверьте сеть без VPN/прокси.');
      setStatusHint('Проверьте интернет: модели скачиваются при первом запуске из Hugging Face. Без сети/при блокировке загрузка не завершится.');
    }
  };

  const handleDownloadZip = async () => {
    if (!transcript || !file) return;

    const zip = new JSZip();
    const baseName = file.name.replace(/\.[^.]+$/, '');

    zip.file(`${baseName}.txt`, transcript.text.trim());
    if (transcript.chunks && transcript.chunks.length > 0) {
      zip.file(`${baseName}.srt`, toSrt(transcript.chunks));
    }

    const content = await zip.generateAsync({ type: 'blob' });
    saveAs(content, `${baseName}-transcript.zip`);
  };

  return (
    <main className="app">
      <section className="panel">
        <h1>OpenTranscribe RU</h1>
        <p className="lead">Локальная транскрибация аудио и видео в браузере без отправки на сервер.</p>

        <fieldset className="field model-radio-group">
          <legend>Модель Whisper</legend>
          {MODELS.map((model) => (
            <label key={model.value} className="radio-item">
              <input
                type="radio"
                name="whisper-model"
                value={model.value}
                checked={selectedModel === model.value}
                onChange={() => setSelectedModel(model.value)}
              />
              <span>{model.label}</span>
            </label>
          ))}
        </fieldset>

        <label className="field">
          <span>Устройство обработки</span>
          <select value={selectedDevice} onChange={(event) => setSelectedDevice(event.target.value as Device)}>
            {DEVICES.map((device) => (
              <option key={device.value} value={device.value}>
                {device.label}
              </option>
            ))}
          </select>
          <small className="hint">{DEVICES.find((item) => item.value === selectedDevice)?.hint}</small>
        </label>

        <label className="field">
          <span>Аудио/видео файл</span>
          <input
            type="file"
            accept="audio/*,video/*"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
        </label>

        <button onClick={handleTranscribe} disabled={!canTranscribe}>
          Транскрибировать
        </button>

        <div className="progress-wrap" aria-hidden={statusPercent === null}>
          <progress value={statusPercent ?? 0} max={100} />
          <span>{statusPercent !== null ? `${statusPercent}%` : '—'}</span>
        </div>

        <p className={`status ${status}`}>{statusText}</p>
        {statusHint && <p className="status-hint">{statusHint}</p>}
      </section>

      <section className="panel">
        <div className="result-header">
          <h2>Результат</h2>
          <button onClick={handleDownloadZip} disabled={!transcript}>Скачать TXT/SRT (ZIP)</button>
        </div>
        <textarea value={transcript?.text ?? ''} readOnly placeholder="Тут появится текст после обработки файла." />
      </section>
    </main>
  );
};

export default App;
