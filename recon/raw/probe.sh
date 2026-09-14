#!/bin/bash
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
for u in "$@"; do
  out=$(curl -sL --max-time 40 --retry 2 --retry-delay 1 --retry-all-errors -A "$UA" -o /tmp/probe_body.$$ -w "%{http_code} %{size_download}" "$u")
  echo "$out  $u"
  if [ "${out%% *}" = "200" ]; then cp /tmp/probe_body.$$ "$(echo "$u" | md5).body"; echo "     saved: $(echo "$u" | md5).body"; fi
done
rm -f /tmp/probe_body.$$
