import json, sys, collections

src = sys.argv[1]                      # path to the big BDD100K labels json
d = json.load(open(src))
print(len(d), "entries; first one:")
print(json.dumps(d[0])[:400])

slim = []
for x in d:
    a = x.get("attributes", {})
    c = collections.Counter(l["category"] for l in (x.get("labels") or []))
    slim.append({"name": x["name"], "weather": a.get("weather"),
                 "scene": a.get("scene"), "timeofday": a.get("timeofday"),
                 "objects": dict(c)})

json.dump(slim, open("labels_slim.json", "w"))
for k in ("weather", "scene", "timeofday"):
    print(k, collections.Counter(s[k] for s in slim).most_common())