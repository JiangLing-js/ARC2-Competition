import http.server
import json
from pathlib import Path

d = Path(r"E:\projects\arc_vitarc_ttt_1\experiments\arc_gr48k_pi_stage2_sft_20261022\delivery")
cpu_nb = json.loads((d / "ARC_STAGE2_DEEPSEEK_V4_1_FLASH_DISTILL_AND_DATASET_CPU_20261022.ipynb").read_text("utf-8"))
g4_nb = json.loads((d / "ARC_QWEN38_27B_STAGE2_PI_SFT_G4_20261022.ipynb").read_text("utf-8"))
body = json.dumps({"cpu_code": cpu_nb["cells"][1]["source"], "g4_code": g4_nb["cells"][1]["source"]}).encode("utf-8")


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()


if __name__ == "__main__":
    print("Serving stage2 payload on http://127.0.0.1:8765", flush=True)
    http.server.HTTPServer(("127.0.0.1", 8765), H).serve_forever()
