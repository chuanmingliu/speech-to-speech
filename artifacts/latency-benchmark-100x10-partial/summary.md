# Synthetic latency benchmark summary

- URL: `ws://127.0.0.1:8765/v1/realtime`
- Run state: stopped early at user request
- Fully checkpointed cases: 36
- Cases planned: 100
- Turns planned: 1000
- Turns recorded: 360
- Successful turns: 360
- Failed turns: 0
- Success rate: 100.00%

## Latency metrics
- `asr_final_to_first_assistant_text_ms`: p50=877.903 ms, p95=4663.456 ms, p99=11761.695 ms, mean=1668.454 ms, n=360
- `audio_send_ms`: p50=3766.612 ms, p95=5249.637 ms, p99=6177.566 ms, mean=3877.703 ms, n=360
- `connection_ms`: p50=8.678 ms, p95=27.422 ms, p99=52.094 ms, mean=10.561 ms, n=360
- `first_assistant_text_to_first_audio_ms`: p50=204.642 ms, p95=345.064 ms, p99=521.861 ms, mean=207.399 ms, n=360
- `first_audio_to_audio_done_ms`: p50=564.570 ms, p95=1099.265 ms, p99=1374.686 ms, mean=620.164 ms, n=360
- `input_audio_end_to_speech_stop_ms`: p50=100.363 ms, p95=139.434 ms, p99=150.041 ms, mean=-496.875 ms, n=360
- `input_audio_ms`: p50=3239.783 ms, p95=4724.732 ms, p99=5650.713 ms, mean=3352.273 ms, n=360
- `speech_start_to_first_asr_partial_ms`: p50=518.880 ms, p95=819.052 ms, p99=2286.082 ms, mean=588.636 ms, n=346
- `speech_stop_to_asr_final_ms`: p50=118.820 ms, p95=1725.928 ms, p99=2649.725 ms, mean=273.581 ms, n=360
- `speech_stop_to_first_audio_ms`: p50=1262.191 ms, p95=4873.588 ms, p99=13819.482 ms, mean=2149.434 ms, n=360
- `speech_stop_to_response_done_ms`: p50=1879.959 ms, p95=5759.666 ms, p99=14300.749 ms, mean=2770.704 ms, n=360
- `turn_start_to_speech_start_ms`: p50=421.423 ms, p95=487.810 ms, p99=497.523 ms, mean=425.523 ms, n=360
- `turn_total_ms`: p50=5137.302 ms, p95=7669.634 ms, p99=16474.544 ms, mean=5626.102 ms, n=360
