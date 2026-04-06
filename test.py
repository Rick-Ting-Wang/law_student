import http.client, json

conn = http.client.HTTPSConnection('dpapi.cn')
payload = json.dumps({
    "model": "deepseek-v3",
    "messages": [{"role": "user", "content": "你好"}]
})
headers = {
    'Authorization': 'Bearer sk-aZ4KJUKxqQGI0kvJBf87Aa2eE53a487e81A8D26f89CfD486',
    'Content-Type': 'application/json'
}
conn.request("POST", "/v1/chat/completions", payload, headers)
res = conn.getresponse()
print(res.read().decode('utf-8'))