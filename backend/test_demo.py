import urllib.request, json

req = urllib.request.Request(
    "http://localhost:8000/api/demo-analyze",
    method="POST",
    headers={"Content-Length": "0"}
)
resp = urllib.request.urlopen(req, timeout=60)
data = json.loads(resp.read().decode("utf-8"))
print("=" * 50)
print("分析引擎:", data.get("engine"))
print("检查部位:", data.get("body_part"))
print("风险级别:", data.get("detection", {}).get("risk_level"))
print("AI置信度:", data.get("detection", {}).get("confidence"))
print("发现异常:", data.get("detection", {}).get("has_significant_finding"))
print("-" * 50)
print("报告摘要（前500字）:")
print(data.get("ai_report", "")[:500])
print("=" * 50)
