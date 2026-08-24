import {
  Button,
  Callout,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type ScenarioId = "clean" | "reopen" | "barge";
type PageId = "timeline" | "ga";
type Vs = "same" | "differ" | "omit" | "ours";

type Lane = {
  who: string;
  event: string;
  detail: string;
};

type Beat = {
  t: string;
  title: string;
  hear: string;
  caption: string;
  spoken: string[];
  asr: Lane | null;
  llm: Lane | null;
  tts: Lane | null;
  vs: Vs;
  ga: string;
  client: string;
};

type Scenario = {
  id: ScenarioId;
  title: string;
  hook: string;
  beats: Beat[];
};

const VS_LABEL: Record<Vs, string> = {
  same: "same as GA",
  differ: "same name, different meaning",
  omit: "GA emits extra events",
  ours: "this cascade only",
};

const CLEAN: Beat[] = [
  {
    t: "0.00s",
    title: "Mic open",
    hear: "PCM is already flowing. No user item yet.",
    caption: "",
    spoken: [],
    asr: {
      who: "Client → AudioHandler → VAD",
      event: "input_audio_buffer.append",
      detail: "Base64 PCM → 16 kHz / 512-sample chunks. WebRTC: RTP, append is rejected.",
    },
    llm: null,
    tts: null,
    vs: "same",
    ga: "Same client event on WebSocket. GA WebRTC also uses the media track and rejects append.",
    client: "Keep streaming. Do not wait for a start ack.",
  },
  {
    t: "0.42s",
    title: "Speech starts",
    hear: "You start: “What's the weather…”",
    caption: "",
    spoken: [],
    asr: {
      who: "VAD → AudioHandler",
      event: "input_audio_buffer.speech_started",
      detail: "Allocates item_u1. Hidden: turn_id=t1 rev=0, Tencent voice_id=voice_A.",
    },
    llm: null,
    tts: null,
    vs: "omit",
    ga: "Same speech_started. GA also emits conversation.item.created (and often conversation.item.added) for the user audio item. We do not — item_id only appears on speech_* / transcription.*.",
    client: "Open a caption slot keyed by item_u1.",
  },
  {
    t: "0.80s",
    title: "First live caption",
    hear: "Caption: “What's”",
    caption: "What's",
    spoken: [],
    asr: {
      who: "Tencent ASR → ConversationHandler",
      event: "conversation.item.input_audio_transcription.delta",
      detail: "Hidden slice_type=1. Wire delta is the FULL current hypothesis. content_index=0.",
    },
    llm: null,
    tts: null,
    vs: "differ",
    ga: "Same event name. Official GA example delta is incremental (“Hello,”) — clients APPEND. Ours is a full rewrite — clients REPLACE. We also increment content_index per snapshot; GA usually keeps content_index=0.",
    client: "REPLACE the caption. Do not concatenate.",
  },
  {
    t: "1.90s",
    title: "Caption revises",
    hear: "Caption: “What's the weather”",
    caption: "What's the weather",
    spoken: [],
    asr: {
      who: "Tencent ASR → ConversationHandler",
      event: "conversation.item.input_audio_transcription.delta",
      detail: "Hidden slice_type=2 (stable) looks identical on the wire. content_index=2. Still a full string.",
    },
    llm: null,
    tts: null,
    vs: "differ",
    ga: "GA has no slice_type. Their deltas stay incremental. A client written for GA APPEND will double every word against this server.",
    client: "REPLACE again. content_index is a snapshot counter, not “next syllable”.",
  },
  {
    t: "2.62s",
    title: "You stop",
    hear: "Silence. Caption is still provisional.",
    caption: "What's the weather in Beijing?",
    spoken: [],
    asr: {
      who: "VAD → AudioHandler",
      event: "input_audio_buffer.speech_stopped",
      detail: "Soft-final. SpeculativeTurnTracker can still reopen t1 until TTS commit.",
    },
    llm: null,
    tts: null,
    vs: "omit",
    ga: "Same speech_stopped. GA server_vad then emits input_audio_buffer.committed and conversation.item.done. We accept commit from the client but emit no committed event.",
    client: "Stop the input meter. Wait for completed before freezing the caption.",
  },
  {
    t: "2.70s",
    title: "Final ASR starts the LLM",
    hear: "Caption locks. You did not send response.create.",
    caption: "What's the weather in Beijing?",
    spoken: [],
    asr: {
      who: "Tencent final → ConversationHandler",
      event: "conversation.item.input_audio_transcription.completed",
      detail: "content_index always 0. Hidden {type:end} / final=1 never leak.",
    },
    llm: {
      who: "TranscriptionNotifier → DeepSeek",
      event: "(no wire) GenerateResponseRequest",
      detail: "Internal queue put. This is the VAD-path generate trigger.",
    },
    tts: null,
    vs: "differ",
    ga: "Same completed event (GA may add languages[]). Implicit generate on server_vad is similar. Difference: our completed can fire again on the same item_id after a reopen. GA treats each committed item as one-shot.",
    client: "Freeze caption. Do not send response.create. Wait for response.created or first output.",
  },
  {
    t: "3.15s",
    title: "First LLM sentence",
    hear: "Subtitle appears. No sound yet.",
    caption: "What's the weather in Beijing?",
    spoken: ["It's twenty-two degrees."],
    asr: null,
    llm: {
      who: "DeepSeek → LMOutputProcessor → ResponseHandler",
      event: "response.created  +  response.output_audio_transcript.done",
      detail: "One .done per flushed sentence, text = that sentence only. stream_batch_sentences=1.",
    },
    tts: {
      who: "SpeculativeTurnTracker + MiniMax",
      event: "(no wire) commit(t1,0)  ·  POST /v1/t2a_v2",
      detail: "Commit locks reopen. One HTTP SSE per TTSInput, not the official T2A WebSocket.",
    },
    vs: "differ",
    ga: "GA emits response.created when generation starts, then response.output_item.added, content_part.added, and response.output_audio_transcript.delta (token stream), then one .done with the FULL transcript. We skip item/part lifecycle, skip transcript.delta, and fire .done per sentence.",
    client: "APPEND the sentence. Start a resp_1 bucket. MiniMax has no captions.",
  },
  {
    t: "3.22s",
    title: "First PCM",
    hear: "Assistant starts speaking.",
    caption: "What's the weather in Beijing?",
    spoken: ["It's twenty-two degrees."],
    asr: null,
    llm: null,
    tts: {
      who: "MiniMax SSE → AudioHandler",
      event: "response.output_audio.delta",
      detail: "Hidden hex PCM → base64 PCM16. content_index=0. First frame may be short.",
    },
    vs: "same",
    ga: "Same event and PLAY rule. Source differs: GA audio is the speech model; ours is MiniMax. On the VAD path we often emit response.created here (first audio) if it did not already fire on the sentence.",
    client: "Decode PCM16 at the negotiated rate. Play immediately. Not MiniMax hex, not mp3.",
  },
  {
    t: "3.85s",
    title: "Second sentence, new TTS POST",
    hear: "Second subtitle while the first sentence is still playing.",
    caption: "What's the weather in Beijing?",
    spoken: ["It's twenty-two degrees.", "Want the weekend forecast?"],
    asr: null,
    llm: {
      who: "DeepSeek flush 2 → ResponseHandler",
      event: "response.output_audio_transcript.done",
      detail: "Same resp_1 / item_a1. Payload is only the new sentence.",
    },
    tts: {
      who: "MiniMax",
      event: "(no wire) new HTTP stream",
      detail: "One HTTP connection per sentence. Not task_continue on a T2A WebSocket.",
    },
    vs: "differ",
    ga: "GA would still be streaming output_audio_transcript.delta tokens into one part, and would not open a second synthesizer. A second .done on the same response is our sentence pipeline, not GA.",
    client: "APPEND the second sentence. Do not replace the first.",
  },
  {
    t: "4.20s",
    title: "Second sentence audio",
    hear: "“Want the weekend forecast?”",
    caption: "What's the weather in Beijing?",
    spoken: ["It's twenty-two degrees.", "Want the weekend forecast?"],
    asr: null,
    llm: null,
    tts: {
      who: "MiniMax SSE → AudioHandler",
      event: "response.output_audio.delta",
      detail: "content_index increments. Same response_id. Concatenate PCM.",
    },
    vs: "same",
    ga: "Same PLAY/append rule for audio deltas. Opposite of ASR deltas on this server.",
    client: "Queue the next PCM batch. Output content_index means append.",
  },
  {
    t: "5.20s",
    title: "Response closes",
    hear: "Playback ends.",
    caption: "What's the weather in Beijing?",
    spoken: ["It's twenty-two degrees.", "Want the weekend forecast?"],
    asr: null,
    llm: null,
    tts: {
      who: "TTS sentinel → ResponseHandler",
      event: "response.output_audio.done  +  response.done",
      detail: "AUDIO_RESPONSE_DONE. status=completed. should_listen stays on.",
    },
    vs: "omit",
    ga: "GA wraps audio with content_part.done and output_item.done before response.done. We jump straight to output_audio.done + response.done.",
    client: "Mark resp_1 finished. Next speech_started is a new user item.",
  },
];

const REOPEN: Beat[] = [
  {
    t: "1.10s",
    title: "You pause mid-question",
    hear: "“What's the weather…” then silence.",
    caption: "What's the weather",
    spoken: [],
    asr: {
      who: "VAD → AudioHandler",
      event: "input_audio_buffer.speech_stopped",
      detail: "Soft-final. t1 still uncommitted.",
    },
    llm: null,
    tts: null,
    vs: "ours",
    ga: "GA server_vad commit is harder — the item is done. There is no first-class “uncommitted turn, please keep the same item_id”.",
    client: "Do not treat speech_stopped as a new user bubble.",
  },
  {
    t: "1.20s",
    title: "First final, hidden think",
    hear: "Caption locks on the short phrase.",
    caption: "What's the weather",
    spoken: [],
    asr: {
      who: "STT → ConversationHandler",
      event: "conversation.item.input_audio_transcription.completed",
      detail: "item_u1 transcript = “What's the weather”.",
    },
    llm: {
      who: "Notifier → DeepSeek",
      event: "(no wire) GenerateResponseRequest t1/0",
      detail: "Model starts. No response.created until a sentence or first audio.",
    },
    tts: null,
    vs: "differ",
    ga: "GA completed ends that item. A later continuation is a new item plus a new response. We keep thinking speculatively on the same item.",
    client: "Freeze caption. A second completed on item_u1 means revision, not a second question.",
  },
  {
    t: "1.45s",
    title: "You continue — same item",
    hear: "“…in Beijing?”",
    caption: "What's the weather",
    spoken: [],
    asr: {
      who: "VAD → AudioHandler + new Tencent WS",
      event: "input_audio_buffer.speech_started",
      detail: "reopened=true, same item_u1, rev=1, new hidden voice_id=voice_B.",
    },
    llm: {
      who: "DeepSeek t1/0",
      event: "(no wire) still running, now stale",
      detail: "Will be dropped at AssistantText by the latest-revision gate.",
    },
    tts: null,
    vs: "ours",
    ga: "GA speech_started after a committed item allocates a NEW item_id and cancels an active response. We reuse item_u1 and do not emit response.done (nothing was created yet).",
    client: "Keep the same caption slot. This is not barge-in.",
  },
  {
    t: "2.30s",
    title: "Second completed revises the prompt",
    hear: "Full question is now the prompt.",
    caption: "What's the weather in Beijing?",
    spoken: [],
    asr: {
      who: "STT → ConversationHandler",
      event: "conversation.item.input_audio_transcription.completed",
      detail: "Same item_u1. Chat user text is replaced, not appended.",
    },
    llm: {
      who: "Notifier → DeepSeek",
      event: "(no wire) GenerateResponseRequest t1/1",
      detail: "Rev 0 is no longer latest.",
    },
    tts: null,
    vs: "ours",
    ga: "GA does not emit a second completed on the same item_id. Clients that key bubbles by item_id must REPLACE the frozen caption here.",
    client: "Second completed on item_u1 = speculative revision.",
  },
  {
    t: "2.55s",
    title: "Stale LLM sentence dropped",
    hear: "Nothing plays.",
    caption: "What's the weather in Beijing?",
    spoken: [],
    asr: null,
    llm: {
      who: "ResponseHandler + SpeculativeTurnTracker",
      event: "(no wire) drop AssistantText t1/0",
      detail: "Not latest after reopen grace. No response.created.",
    },
    tts: {
      who: "TTS",
      event: "(none)",
      detail: "Never committed, so reopen was still legal.",
    },
    vs: "ours",
    ga: "GA would have already spoken or cancelled on the wire. Swallowing a generation with zero events is cascade-only.",
    client: "Do nothing. Silence is success.",
  },
  {
    t: "3.10s",
    title: "Latest sentence commits and speaks",
    hear: "The real answer starts.",
    caption: "What's the weather in Beijing?",
    spoken: ["It's twenty-two degrees in Beijing."],
    asr: null,
    llm: {
      who: "DeepSeek t1/1 → ResponseHandler",
      event: "response.created  +  output_audio_transcript.done",
      detail: "commit(t1,1). Further speech is barge-in.",
    },
    tts: {
      who: "MiniMax → AudioHandler",
      event: "response.output_audio.delta",
      detail: "Same PLAY rule as the clean turn.",
    },
    vs: "differ",
    ga: "From here the names match GA again. The missing history is the silent rev-0 generate.",
    client: "APPEND subtitle. PLAY PCM.",
  },
];

const BARGE: Beat[] = [
  {
    t: "3.40s",
    title: "TTS is mid-sentence",
    hear: "You hear “It's twenty-two…”",
    caption: "What's the weather in Beijing?",
    spoken: ["It's twenty-two degrees."],
    asr: null,
    llm: null,
    tts: {
      who: "MiniMax → send loop",
      event: "response.output_audio.delta  (in flight)",
      detail: "PCM on output_queue. in_response=true.",
    },
    vs: "same",
    ga: "Same picture: an active response is playing audio deltas.",
    client: "Keep playing until a cancel arrives.",
  },
  {
    t: "3.48s",
    title: "You cut in — text beats audio",
    hear: "“No, I meant Shanghai.” Playback should stop.",
    caption: "What's the weather in Beijing?",
    spoken: ["It's twenty-two degrees."],
    asr: {
      who: "VAD → AudioHandler → send loop",
      event: "input_audio_buffer.speech_started",
      detail: "NEW item_u2. Text queue is drained before the next PCM chunk.",
    },
    llm: {
      who: "ResponseHandler.finish_response",
      event: "response.done  status=cancelled  reason=turn_detected",
      detail: "CancelScope.generation++. Stale LLM tokens die.",
    },
    tts: {
      who: "Send loop",
      event: "(no more deltas)  +  WebRTC discard_pending_audio",
      detail: "output_queue flushed. Hidden MiniMax frames dropped by generation stamp.",
    },
    vs: "differ",
    ga: "Same cancel contract (speech_started then cancelled response). GA docs sometimes say response.cancelled — the GA wire is response.done with status=cancelled. We match that. Difference: we also require the text-before-audio send loop because TTS PCM is a separate queue from VAD events.",
    client: "STOP on speech_started. Treat turn_detected as official cancel.",
  },
  {
    t: "3.90s",
    title: "New ASR on a new item",
    hear: "Old assistant bubble stays truncated.",
    caption: "No I meant Shanghai",
    spoken: ["It's twenty-two degrees."],
    asr: {
      who: "Tencent voice_C → ConversationHandler",
      event: "conversation.item.input_audio_transcription.delta",
      detail: "Writes item_u2, not item_u1. REPLACE rule unchanged.",
    },
    llm: null,
    tts: null,
    vs: "same",
    ga: "Same: barge-in is a new item_id. Contrast with reopen, which reused item_u1.",
    client: "REPLACE caption on item_u2.",
  },
  {
    t: "4.40s",
    title: "New final, new response",
    hear: "A fresh Shanghai answer starts.",
    caption: "No I meant Shanghai",
    spoken: ["It's twenty-two degrees.", "Shanghai is nineteen degrees."],
    asr: {
      who: "STT → ConversationHandler",
      event: "conversation.item.input_audio_transcription.completed",
      detail: "item_u2. Implicit generate again.",
    },
    llm: {
      who: "Notifier → DeepSeek → ResponseHandler",
      event: "response.created  +  output_audio_transcript.done",
      detail: "resp_2 / item_a2. new_response() clears the discard window.",
    },
    tts: {
      who: "MiniMax → AudioHandler",
      event: "response.output_audio.delta",
      detail: "New HTTP stream. Old MiniMax connection is irrelevant.",
    },
    vs: "differ",
    ga: "Names match. We still skip transcript.delta / output_item.added / content_part.*.",
    client: "New response_id. APPEND new sentence. PLAY new PCM.",
  },
];

const SCENARIOS: Scenario[] = [
  {
    id: "clean",
    title: "Clean turn",
    hook: "One question, two sentences. Read left to right: ASR, then LLM, then TTS.",
    beats: CLEAN,
  },
  {
    id: "reopen",
    title: "Soft-final reopen",
    hook: "Pause, hidden think, continue. Same item_id — GA would have made a new one.",
    beats: REOPEN,
  },
  {
    id: "barge",
    title: "Barge-in",
    hook: "Talk over TTS. VAD, ResponseHandler, and the send loop fire on one beat.",
    beats: BARGE,
  },
];

const GA_ROWS: Array<{
  event: string;
  who: string;
  lane: string;
  vs: Vs;
  note: string;
}> = [
  {
    event: "input_audio_buffer.append",
    who: "Client → AudioHandler",
    lane: "ASR",
    vs: "same",
    note: "WebRTC: rejected here and in GA; audio is the media track.",
  },
  {
    event: "input_audio_buffer.speech_started",
    who: "VAD → AudioHandler",
    lane: "ASR",
    vs: "same",
    note: "Allocates item_id. On barge-in, also the cancel signal.",
  },
  {
    event: "input_audio_buffer.speech_stopped",
    who: "VAD → AudioHandler",
    lane: "ASR",
    vs: "same",
    note: "VAD clock, not ASR final.",
  },
  {
    event: "input_audio_buffer.committed",
    who: "—",
    lane: "ASR",
    vs: "omit",
    note: "GA server_vad emits this. We accept client commit and only log.",
  },
  {
    event: "conversation.item.created / added / done",
    who: "ConversationHandler (create only)",
    lane: "ASR / tools",
    vs: "omit",
    note: "We emit item.created only for injected text / tool output. GA emits it for user audio items too. item.added / item.done are GA-only.",
  },
  {
    event: "conversation.item.input_audio_transcription.delta",
    who: "STT → ConversationHandler",
    lane: "ASR",
    vs: "differ",
    note: "GA: incremental token, APPEND, content_index usually 0. Us: full hypothesis, REPLACE, content_index increments.",
  },
  {
    event: "conversation.item.input_audio_transcription.completed",
    who: "STT → ConversationHandler",
    lane: "ASR",
    vs: "differ",
    note: "Same name. Ours always content_index=0 and can fire twice on one item_id after reopen. Also the implicit GenerateResponseRequest.",
  },
  {
    event: "response.created",
    who: "ResponseHandler or AudioHandler",
    lane: "LLM / TTS",
    vs: "differ",
    note: "GA: when generation starts. Us: on response.create, or on the first audio chunk of the VAD path (sometimes after the first transcript.done).",
  },
  {
    event: "response.output_item.added / done",
    who: "—",
    lane: "LLM",
    vs: "omit",
    note: "GA response lifecycle. We never emit these.",
  },
  {
    event: "response.content_part.added / done",
    who: "—",
    lane: "LLM / TTS",
    vs: "omit",
    note: "GA part lifecycle around text/audio. We skip it.",
  },
  {
    event: "response.output_audio_transcript.delta",
    who: "—",
    lane: "LLM",
    vs: "omit",
    note: "GA streams assistant tokens. We never emit this. Do not wait for it.",
  },
  {
    event: "response.output_audio_transcript.done",
    who: "DeepSeek sentence → ResponseHandler",
    lane: "LLM",
    vs: "differ",
    note: "GA: once per response, full transcript. Us: once per LLM sentence, that sentence only. MiniMax is not the source.",
  },
  {
    event: "response.output_audio.delta",
    who: "MiniMax → AudioHandler",
    lane: "TTS",
    vs: "same",
    note: "Base64 PCM16. PLAY / append chunks. Source is MiniMax HTTP SSE, not the OpenAI speech model.",
  },
  {
    event: "response.output_audio.done",
    who: "ResponseHandler.finish_response",
    lane: "TTS",
    vs: "same",
    note: "Closes audio for this response. No PCM in the payload.",
  },
  {
    event: "response.done",
    who: "ResponseHandler",
    lane: "LLM / TTS",
    vs: "same",
    note: "completed, or cancelled with reason turn_detected / client_cancelled. Not the old beta response.cancelled.",
  },
  {
    event: "response.function_call_arguments.done",
    who: "LMOutputProcessor → ResponseHandler",
    lane: "LLM",
    vs: "omit",
    note: "We skip function_call_arguments.delta. GA streams argument tokens first.",
  },
  {
    event: "rate_limits.updated",
    who: "—",
    lane: "—",
    vs: "omit",
    note: "GA only.",
  },
];

function scenarioById(id: ScenarioId): Scenario {
  return SCENARIOS.find((s) => s.id === id) ?? SCENARIOS[0];
}

function VsPill({ vs }: { vs: Vs }) {
  return <Pill active={vs === "differ" || vs === "ours"}>{VS_LABEL[vs]}</Pill>;
}

function LaneCell({ lane, label }: { lane: Lane | null; label: string }) {
  const theme = useHostTheme();
  if (!lane) {
    return (
      <div
        style={{
          padding: 10,
          background: theme.bg.editor,
          border: `1px solid ${theme.stroke.tertiary}`,
          minHeight: 88,
        }}
      >
        <Text size="small" tone="quaternary">
          {label}
        </Text>
        <div style={{ marginTop: 6 }}>
          <Text tone="tertiary">—</Text>
        </div>
      </div>
    );
  }
  return (
    <div
      style={{
        padding: 10,
        background: theme.fill.tertiary,
        border: `1px solid ${theme.stroke.secondary}`,
        minHeight: 88,
      }}
    >
      <Text size="small" tone="tertiary">
        {label}
      </Text>
      <div style={{ marginTop: 4 }}>
        <Text size="small" tone="secondary">
          {lane.who}
        </Text>
      </div>
      <div style={{ marginTop: 6 }}>
        <Text weight="semibold">{lane.event}</Text>
      </div>
      <div style={{ marginTop: 6 }}>
        <Text size="small" tone="secondary">
          {lane.detail}
        </Text>
      </div>
    </div>
  );
}

function TimelinePage({
  scenario,
  beat,
  index,
  setIndex,
}: {
  scenario: Scenario;
  beat: Beat;
  index: number;
  setIndex: (n: number) => void;
}) {
  const theme = useHostTheme();
  const last = scenario.beats.length - 1;
  return (
    <Stack gap={16}>
      <Row gap={8} wrap>
        {scenario.beats.map((b, idx) => (
          <div key={`${scenario.id}-${b.t}-${b.title}`}>
            <Button
              variant={idx === index ? "primary" : "ghost"}
              onClick={() => setIndex(idx)}
            >
              {b.t}
            </Button>
          </div>
        ))}
      </Row>

      <Row gap={8} align="center">
        <Button
          variant="secondary"
          disabled={index === 0}
          onClick={() => setIndex(index - 1)}
        >
          Prev
        </Button>
        <Button
          variant="primary"
          disabled={index === last}
          onClick={() => setIndex(index + 1)}
        >
          Next
        </Button>
        <Text weight="semibold">
          {beat.t} · {beat.title}
        </Text>
        <VsPill vs={beat.vs} />
      </Row>

      <Text italic tone="secondary">
        {beat.hear}
      </Text>

      <Grid columns={3} gap={8}>
        <Stat value={beat.caption || "—"} label="ASR caption (REPLACE)" />
        <Stat
          value={beat.spoken[beat.spoken.length - 1] || "—"}
          label="Latest LLM sentence (APPEND)"
        />
        <Stat
          value={beat.tts?.event.startsWith("response.output_audio.delta") ? "playing" : "idle"}
          label="TTS speaker"
        />
      </Grid>

      <Grid columns={3} gap={8}>
        <LaneCell label="ASR" lane={beat.asr} />
        <LaneCell label="LLM" lane={beat.llm} />
        <LaneCell label="TTS" lane={beat.tts} />
      </Grid>

      <div
        style={{
          padding: 12,
          background: theme.bg.elevated,
          border: `1px solid ${theme.stroke.tertiary}`,
        }}
      >
        <H3>OpenAI GA at this beat</H3>
        <div style={{ marginTop: 8 }}>
          <Text>{beat.ga}</Text>
        </div>
      </div>

      <Callout tone="info" title="Client">
        {beat.client}
      </Callout>
    </Stack>
  );
}

function ContrastPage() {
  return (
    <Stack gap={12}>
      <Text tone="secondary">
        Event names follow OpenAI Realtime GA (not the beta{" "}
        <Code>response.audio.delta</Code> names). Same name does not mean same
        merge rule.
      </Text>
      <Table
        headers={["GA / wire event", "Who says it here", "Lane", "vs GA", "Contrast"]}
        columnAlign={["left", "left", "left", "left", "left"]}
        striped
        stickyHeader
        rows={GA_ROWS.map((r) => [
          r.event,
          r.who,
          r.lane,
          VS_LABEL[r.vs],
          r.note,
        ])}
        rowTone={GA_ROWS.map((r) =>
          r.vs === "same" ? "success" : r.vs === "differ" || r.vs === "ours" ? "warning" : "info",
        )}
      />
    </Stack>
  );
}

export default function StreamingRealtimeCoop() {
  const dispatch = useCanvasAction();
  const [page, setPage] = useCanvasState<PageId>("page", "timeline");
  const [scenarioId, setScenarioId] = useCanvasState<ScenarioId>("scenario", "clean");
  const [beatIndex, setBeatIndex] = useCanvasState("beat", 0);
  const scenario = scenarioById(scenarioId);
  const last = scenario.beats.length - 1;
  const i = beatIndex < 0 ? 0 : beatIndex > last ? last : beatIndex;
  const beat = scenario.beats[i];

  const goScenario = (id: ScenarioId) => {
    setScenarioId(id);
    setBeatIndex(0);
    setPage("timeline");
  };

  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 1080 }}>
      <Stack gap={8}>
        <H1>ASR / LLM / TTS on the Realtime wire</H1>
        <Text tone="secondary">
          Each beat is one moment on the timeline. The three columns are who
          speaks: ASR (VAD + STT), LLM (DeepSeek + notifier), TTS (MiniMax).
          The protocol name is what the client sees. The GA note is what
          OpenAI Realtime would have emitted at the same moment.
        </Text>
      </Stack>

      <Row gap={8} wrap>
        <Button
          variant={page === "timeline" ? "primary" : "secondary"}
          onClick={() => setPage("timeline")}
        >
          Timeline
        </Button>
        <Button
          variant={page === "ga" ? "primary" : "secondary"}
          onClick={() => setPage("ga")}
        >
          vs OpenAI GA
        </Button>
      </Row>

      {page === "timeline" ? (
        <Stack gap={12}>
          <Row gap={8} wrap>
            {SCENARIOS.map((s) => (
              <div key={s.id}>
                <Button
                  variant={s.id === scenarioId ? "primary" : "secondary"}
                  onClick={() => goScenario(s.id)}
                >
                  {s.title}
                </Button>
              </div>
            ))}
          </Row>
          <Text tone="secondary">{scenario.hook}</Text>
          <TimelinePage
            scenario={scenario}
            beat={beat}
            index={i}
            setIndex={setBeatIndex}
          />
        </Stack>
      ) : (
        <ContrastPage />
      )}

      <Divider />

      <H2>Client verbs that disagree with GA</H2>
      <Grid columns={2} gap={12}>
        <Text>
          <Text weight="semibold">ASR delta → REPLACE.</Text> Official GA
          transcription examples APPEND incremental tokens. This server sends
          the full current hypothesis on every{" "}
          <Code>input_audio_transcription.delta</Code>.
        </Text>
        <Text>
          <Text weight="semibold">TTS transcript → APPEND per sentence.</Text>{" "}
          GA streams <Code>output_audio_transcript.delta</Code> then one{" "}
          <Code>.done</Code>. We never emit <Code>.delta</Code>. Each{" "}
          <Code>.done</Code> is one DeepSeek flush.
        </Text>
      </Grid>

      <H2>Where the adapter speaks</H2>
      <Row gap={8} wrap>
        <Button
          variant="secondary"
          onClick={() =>
            dispatch({
              type: "openFile",
              path: "src/speech_to_speech/api/openai_realtime/handlers/conversation.py",
              selection: { startLine: 100, startColumn: 1 },
            })
          }
        >
          ASR protocol
        </Button>
        <Button
          variant="secondary"
          onClick={() =>
            dispatch({
              type: "openFile",
              path: "src/speech_to_speech/api/openai_realtime/handlers/response.py",
              selection: { startLine: 263, startColumn: 1 },
            })
          }
        >
          LLM protocol
        </Button>
        <Button
          variant="secondary"
          onClick={() =>
            dispatch({
              type: "openFile",
              path: "src/speech_to_speech/api/openai_realtime/handlers/audio.py",
              selection: { startLine: 108, startColumn: 1 },
            })
          }
        >
          TTS + VAD protocol
        </Button>
        <Button
          variant="secondary"
          onClick={() =>
            dispatch({
              type: "openFile",
              path: "src/speech_to_speech/api/openai_realtime/README.md",
              selection: { startLine: 54, startColumn: 1 },
            })
          }
        >
          Event list
        </Button>
      </Row>
    </Stack>
  );
}
