import { useMemo, useState } from 'react';
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

const MODELS = [
  { value: 'Xenova/whisper-tiny', label: 'Whisper Tiny (быстро)' },
  { value: 'Xenova/whisper-base', label: 'Whisper Base (точнее)' },
] as const;

let currentModelName = '';
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

const App = () => {
  const [selectedModel, setSelectedModel] = useState<(typeof MODELS)[number]['value']>(MODELS[0].value);
  const [status, setStatus] = useState<Status>('idle');
  const [statusText, setStatusText] = useState('Выберите файл и нажмите «Транскрибировать».');
  const [file, setFile] = useState<File | null>(null);
  const [transcript, setTranscript] = useState<TranscriptResult | null>(null);

  const canTranscribe = useMemo(() => !!file && (status === 'idle' || status === 'done' || status === 'error'), [file, status]);

  const loadPipeline = async () => {
    if (currentAsr && currentModelName === selectedModel) return currentAsr;

    setStatus('loading-model');
    setStatusText('Загружаем модель. Это может занять несколько минут при первом запуске.');

    currentAsr = await pipeline('automatic-speech-recognition', selectedModel, {
      progress_callback: (progress: { file?: string; progress?: number }) => {
        if (!progress) return;
        const part = progress.file ?? 'модель';
        const percentage = progress.progress ? ` (${Math.round(progress.progress)}%)` : '';
        setStatusText(`Загрузка: ${part}${percentage}`);
      },
    });
    currentModelName = selectedModel;
    return currentAsr;
  };

  const handleTranscribe = async () => {
    if (!file) return;

    setTranscript(null);
    setStatus('transcribing');
    setStatusText('Идёт распознавание речи...');

    try {
      const asr = await loadPipeline();
      const result = (await (asr as any)(file, {
        chunk_length_s: 30,
        stride_length_s: 5,
        return_timestamps: true,
        language: 'russian',
        task: 'transcribe',
      })) as TranscriptResult;

      setTranscript(result);
      setStatus('done');
      setStatusText('Готово! Можно скопировать текст или скачать файлы.');
    } catch (error) {
      console.error(error);
      setStatus('error');
      setStatusText('Ошибка транскрибации. Проверьте консоль браузера и попробуйте другой файл.');
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

        <label className="field">
          <span>Модель Whisper</span>
          <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value as (typeof MODELS)[number]['value'])}>
            {MODELS.map((model) => (
              <option key={model.value} value={model.value}>
                {model.label}
              </option>
            ))}
          </select>
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

        <p className={`status ${status}`}>{statusText}</p>
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
