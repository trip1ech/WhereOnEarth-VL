#!/bin/zsh
set -e
tool_dir="${0:A:h}"
url="http://127.0.0.1:8766/"
if curl --max-time 2 -fsS "$url" | grep -Fq '<title>Road-Marking Annotator · WhereOnEarth-VL</title>'; then
    open "$url"
else
    (
        for attempt in {1..20}; do
            if curl --max-time 2 -fsS "$url" 2>/dev/null | grep -Fq '<title>Road-Marking Annotator · WhereOnEarth-VL</title>'; then
                open "$url"
                break
            fi
            sleep 0.25
        done
    ) &
    exec python3 "$tool_dir/server.py"
fi
