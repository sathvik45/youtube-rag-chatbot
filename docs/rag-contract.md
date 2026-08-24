# RAG Engine Contract

## 1. Purpose

The RAG engine answers questions only from the transcripts of YouTube videos selected for the current chat.

It must return either:

1. A cited answer grounded in retrieved transcript chunks, or
2. A controlled refusal when the selected videos do not contain enough information.

The engine must not use outside knowledge to answer video-specific questions.

## 2. Inputs

Every RAG request requires:

```json
{
  "thread_id": "unique-chat-id",
  "question": "What does the speaker say about JWT expiration?",
  "video_ids": ["FqR5vESuKe0"],
  "history": [
    {
      "role": "user",
      "content": "What authentication method does the speaker recommend?"
    },
    {
      "role": "assistant",
      "content": "The speaker recommends..."
    }
  ],
  "request_id": "unique-request-id"
}
```

### Input rules

- `question` must be non-empty.
- `video_ids` must contain at least one canonical YouTube video ID.
- The engine must search only within the supplied `video_ids`.
- A playlist URL is never passed to retrieval; it must first be expanded into video IDs.
- `history` contains only recent user and assistant messages.
- Conversation history helps interpret follow-up questions but is not evidence for an answer.
- Transcript chunks retrieved from the approved video IDs are the only factual source.

## 3. Processing flow

```text
Validate input
→ rewrite follow-up question when history exists
→ retrieve chunks only from approved video IDs
→ refuse if no chunks are retrieved
→ generate answer from retrieved chunks only
→ resolve citation IDs into YouTube timestamp links
→ verify each quoted citation against its source chunk
→ return result
```

## 4. Retrieval contract

### Retrieval input

```json
{
  "query": "standalone rewritten question",
  "video_ids": ["FqR5vESuKe0"],
  "k": 5
}
```

### Retrieval output

Every retrieved chunk must contain:

```json
{
  "text": "Transcript text for this timestamp range.",
  "video_id": "FqR5vESuKe0",
  "video_title": "Video title",
  "chunk_index": 12,
  "start_ms": 184000,
  "end_ms": 201000
}
```

### Retrieval guarantees

- Every returned chunk belongs to an approved `video_id`.
- Empty retrieval is a valid result, not a system exception.
- Retrieved chunks have non-empty text.
- Each chunk has valid timestamps where `start_ms <= end_ms`.
- Retrieval settings are recorded: embedding model, chunking settings, `k`, score threshold, and per-video cap.

## 5. Answer-result contract

Every request returns one of these statuses:

- `answered`
- `refused_no_context`
- `refused_insufficient_context`
- `temporary_error`
- `internal_error`

### Successful answer

```json
{
  "status": "answered",
  "answer": "The speaker recommends short-lived access tokens. [c1: \"access tokens should be short lived\"]",
  "display_answer": "The speaker recommends short-lived access tokens.",
  "citations": [],
  "grounded": true,
  "retrieval": {
    "query_used": "What does the speaker say about JWT expiration?",
    "chunks_retrieved": 3,
    "videos_searched": 1
  },
  "request_id": "unique-request-id"
}
```

### Refusal

```json
{
  "status": "refused_no_context",
  "answer": "I couldn't find anything about that in this video. Try rephrasing, or ask about something the video actually covers.",
  "display_answer": "I couldn't find anything about that in this video.",
  "citations": [],
  "grounded": null,
  "request_id": "unique-request-id"
}
```

## 6. Citation contract

Every citation includes:

```json
{
  "video_id": "FqR5vESuKe0",
  "title": "Video title",
  "start_ms": 184000,
  "end_ms": 201000,
  "label": "03:04",
  "url": "https://www.youtube.com/watch?v=FqR5vESuKe0&t=181s",
  "quotes": ["access tokens should be short lived"],
  "verified": true,
  "score": 1.0
}
```

### Citation policy

- `verified: true` means the quoted text was found in the cited transcript chunk.
- `verified: false` means the model cited a real chunk but altered or invented the quoted text.
- `verified: null` means there was no quote to verify.
- A normal answer should include at least one citation.
- An answer with failed citations must be logged as ungrounded.
- The system must never invent timestamps; timestamp links are computed from chunk metadata.

## 7. Failure behavior

| Situation | Status | Expected behavior |
|---|---|---|
| No relevant chunks retrieved | `refused_no_context` | Do not call the answer generator |
| Chunks do not answer the question | `refused_insufficient_context` | Return a clear refusal |
| LLM, embedding, or vector provider times out | `temporary_error` | Safe retry message; log details |
| Invalid RAG input or unexpected bug | `internal_error` | Do not expose internal details to user |

## 8. Conversation-memory policy

- Store full chat history later in PostgreSQL.
- Send only a bounded recent history window to LangGraph.
- Use history only to rewrite ambiguous follow-up questions.
- Retrieved chunks remain the factual source of truth.
- Each conversation has one fixed scope of approved video IDs.

## 9. Observability requirements

For every request, record:

```text
request_id
thread_id
original question
rewritten question
approved video IDs
number of chunks retrieved
citation verification result
retrieval time
generation time
total time
LLM model
embedding model
prompt version
final status
```

## 10. Acceptance criteria

The RAG engine is stable when:

1. It answers only from selected video IDs.
2. It refuses safely when no evidence is found.
3. Every normal answer has usable timestamp citations.
4. Citation verification failures are visible and logged.
5. The same test set can be rerun after any chunking, embedding, retrieval, or prompt change.
6. External failures produce controlled statuses instead of crashes.