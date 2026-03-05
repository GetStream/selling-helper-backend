# Workflow Events Contract

This document defines the backend-to-frontend custom events sent through Stream Chat channels for the product workflow.

## Transport

- Backend emits Stream custom events using `channel.send_event(payload, server_user_id)`.
- Channel strategy: one channel per workflow call.
- Channel type: `messaging` by default (configurable via `STREAM_WORKFLOW_CHANNEL_TYPE`).
- Channel id format: `workflow-{call_id}` (sanitized for Stream-compatible IDs).

## Envelope Conventions

- All events include a top-level `type`.
- Field naming is `camelCase`.
- Time fields are ISO-8601 UTC strings.

## Event List

### `workflow-started`

Triggered right before orchestration begins.

Payload:

- `type`: `"workflow-started"`
- `workflowId`: string (`call_id`)
- `channelType`: string
- `channelId`: string
- `startedAt`: ISO-8601 UTC timestamp

Example:

```json
{
  "type": "workflow-started",
  "workflowId": "my-call-id",
  "channelType": "messaging",
  "channelId": "workflow-my-call-id",
  "startedAt": "2026-03-05T12:40:00.123456+00:00"
}
```

### `workflow-task-progress`

Triggered for progress updates emitted by the orchestrator task steps.

Payload:

- `type`: `"workflow-task-progress"`
- `workflowId`: string (`call_id`)
- `taskId`: string (`detect-object`, `capture-screenshot`, `polish-image`, `research-product`, `generate-description`)
- `progress`: integer (`0-100`)
- `message`: string (human-readable status)
- `state`: `"running"` or `"completed"` (future-compatible with `"failed"`)
- `timestamp`: ISO-8601 UTC timestamp

Example:

```json
{
  "type": "workflow-task-progress",
  "workflowId": "my-call-id",
  "taskId": "polish-image",
  "progress": 100,
  "message": "Polished product image created",
  "state": "completed",
  "timestamp": "2026-03-05T12:41:18.654321+00:00"
}
```

### `workflow-completed`

Triggered once after a successful workflow run.

Payload:

- `type`: `"workflow-completed"`
- `workflowId`: string (`call_id`)
- `completedAt`: ISO-8601 UTC timestamp
- `resultSummary`: optional string

Example:

```json
{
  "type": "workflow-completed",
  "workflowId": "my-call-id",
  "completedAt": "2026-03-05T12:42:10.000000+00:00",
  "resultSummary": "Espresso machine (0.84)"
}
```

### `workflow-failed`

Triggered when orchestration raises an unhandled exception.

Payload:

- `type`: `"workflow-failed"`
- `workflowId`: string (`call_id`)
- `failedAt`: ISO-8601 UTC timestamp
- `errorCode`: string (exception class name)
- `errorMessage`: string
- `taskId`: optional string (if known)

Example:

```json
{
  "type": "workflow-failed",
  "workflowId": "my-call-id",
  "failedAt": "2026-03-05T12:42:50.000000+00:00",
  "errorCode": "TimeoutError",
  "errorMessage": "Object was not detected clearly enough within timeout window."
}
```

## FE Consumption Notes

- Listen by Stream event `type`.
- Events are emitted in-order by awaiting each send.
- Delivery is best-effort if Stream send fails in non-strict mode.
