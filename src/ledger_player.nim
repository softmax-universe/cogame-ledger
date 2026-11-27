## Ledger model, prompt, and scripted player.
##
## Modes:
## - PLAYER_SCRIPTED=mirror|shark: registers the built-in scripted baseline.
## - COWORLD_LLM_ENABLED=true (or an LLM transport env such as
##   COWORLD_LLM_ENDPOINT / ANTHROPIC_API_KEY in this container): registers
##   external control over `ledger.player.v2`, receives private round
##   observations, calls the routed LLM directly from this container (one call
##   per round, no retries, no scripted fallback), and submits validated
##   actions. Any missing credentials, transport failure, or unparseable reply
##   exits non-zero so the game terminates the episode via player failure.
## - Legacy server-side prompt mode (when neither PLAYER_SCRIPTED nor learner
##   container LLM routing is configured): delivers PLAYER_PROMPT (or the
##   default strategy) to the game server.
##
## To field your own model policy, reuse this image and set PLAYER_PROMPT:
##   coworld upload-policy <ledger-image> --name my-ledger \
##     --run /bin/ledger-player \
##     --secret-env COWORLD_LLM_ENABLED=true \
##     --secret-env COWORLD_LLM_MODEL=anthropic/claude-haiku-4.5 \
##     --secret-env PLAYER_PROMPT="<your strategy>"

import
  std/[json, options, os, strutils],
  whisky,
  ledger/[llm, types]

const DefaultPrompt = """
Your score is the MEDIAN of your per-meeting payoffs, so aim to be reliably
worth meeting rather than spectacularly lucky once. Read the partner record
before every meeting. Open cooperatively with any alias whose record is clean
or empty: cooperate in DILEMMA, send 5 or 6 as investor, return 50 percent as
trustee, offer 6 as proposer. Against an alias whose record shows defections,
zero returns, or lowball offers, defect, send 0 or 1, return 0 to 25 percent,
and keep your responder floor at 5. As responder, set a floor of 3 or 4 in
normal play: high enough to deter lowballs, low enough that most offers clear
it, because a broken deal pays you nothing and a nothing drags your median
down. Keep any optional note or memo to brief plain text with no markdown or
backticks.
"""

proc scriptedName(): string =
  let raw = getEnv("PLAYER_SCRIPTED").strip()
  if raw.len == 0:
    return ""
  let lower = raw.toLowerAscii()
  if lower == "mirror" or lower == "shark":
    return lower
  quit("unknown PLAYER_SCRIPTED baseline: " & raw &
    " (expected mirror or shark)", 1)

proc useContainerLlmMode(scripted: string): bool =
  if scripted.len > 0:
    return false
  let enabled = getEnv("COWORLD_LLM_ENABLED").strip().toLowerAscii()
  if enabled in ["true", "1", "yes"]:
    return true
  for key in [
    "COWORLD_LLM_ENDPOINT",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY_URI",
    "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
    "AWS_BEARER_TOKEN_BEDROCK"
  ]:
    if getEnv(key).strip().len > 0:
      return true
  false

when isMainModule:
  let url = getEnv("COWORLD_PLAYER_WS_URL")
  if url.len == 0:
    quit("COWORLD_PLAYER_WS_URL is not set", 1)
  let scripted = scriptedName()
  let containerModelMode = useContainerLlmMode(scripted)
  var prompt =
    if scripted.len > 0: ""
    else: getEnv("PLAYER_PROMPT")
  if scripted.len == 0 and prompt.len == 0:
    prompt = DefaultPrompt

  proc promptFrame(): string =
    $ %*{"type": "prompt", "prompt": prompt, "scripted": scripted}

  echo "ledger player: connecting to game"
  let socket = newWebSocket(url)
  var slot = -1
  var modelConfig = defaultGameConfig()
  var llmClient: LlmClient = nil

  ## whisky's receiveMessage RAISES on a close frame or a truncated read
  ## (only a timeout returns none), and mummy's send only QUEUES, so the
  ## game's quit(0) can outrun the final frame. In scripted/legacy modes an
  ## unexpected close is normal; in container model mode any LLM/action error
  ## exits 1 explicitly before closing so the game fails the episode.
  var exitCode = 0
  try:
    while true:
      let received = socket.receiveMessage()
      if received.isNone:
        echo "ledger player: connection closed, exiting"
        break
      let message = received.get()
      if message.kind != TextMessage:
        continue
      let payload = parseJson(message.data)
      case payload{"type"}.getStr()
      of "welcome":
        slot = payload{"slot"}.getInt(-1)
        echo "ledger player: seated at slot ", slot,
          " as ", payload{"name"}.getStr()
        if containerModelMode:
          modelConfig.maxOutputTokens = payload{"maxOutputTokens"}.getInt(modelConfig.maxOutputTokens)
          modelConfig.llmTimeoutSeconds = payload{"llmTimeoutSeconds"}.getInt(modelConfig.llmTimeoutSeconds)
          socket.send($ %*{"type": "register", "control": "external"})
          echo "ledger player: registered external model control"
        else:
          socket.send(promptFrame())
          echo "ledger player: profile delivered (", prompt.len, " chars",
            (if scripted.len > 0: ", scripted " & scripted else: ""), ")"
      of "observation":
        if containerModelMode:
          let roundNum = payload["round"].getInt()
          try:
            if llmClient.isNil:
              llmClient = newPlayerLlmClient(modelConfig)
            let decision = chooseExternalModelAction(
              llmClient, payload["observation"], prompt, slot
            )
            socket.send($ %*{
              "type": "action",
              "round": roundNum,
              "action": {
                "move": decision.move,
                "note": decision.note,
                "memo": decision.memo
              }
            })
          except CatchableError as error:
            echo "ledger player: round ", roundNum + 1,
              " model decision failed: ", error.msg
            try:
              socket.send($ %*{
                "type": "action_error",
                "round": roundNum,
                "error": error.msg
              })
              sleep(100)
            except CatchableError:
              discard
            exitCode = 1
            break
      of "final":
        echo "ledger player: final scores ", payload{"scores"}
        break
      else:
        discard
  except CatchableError as error:
    echo "ledger player: socket closed (", error.msg, "); exiting ", exitCode
  try:
    socket.close()
  except CatchableError:
    discard
  quit(exitCode)
