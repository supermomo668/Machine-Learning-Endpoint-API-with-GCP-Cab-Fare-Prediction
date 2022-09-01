# Locally

curl 'localhost:5000/textToSpeech?text=I+would+like+to+go+from+Chelsea+Market+to+The+Juilliard+School.'
cat the_cooper_union_the_juilliard_school.wav | base64 | curl -s -H "Content-Type: application/octet-stream" --data @- localhost:5000/speechToText | jq

curl 'localhost:5000/directions?origin=Brooklyn+Botanic+Garden&destination=Chelsea+Market'

curl 'localhost:5000/namedEntities?text=Flatiron+Building+and+Brooklyn+Bridge'

# curl -X POST 'localhost:5000/predict'
curl -d '[{"pickup_datetime":"2015-05-20 01:40:28 UTC","pickup_longitude":-73.9714126587,"pickup_latitude":40.7604141235,"dropoff_longitude":-73.9626083374,"dropoff_latitude":40.7623062134,"passenger_count":2}]' -H 'Content-Type: application/json' 'localhost:5000/predict'
#curl -X POST 'localhost:5000/farePrediction'
cat the_cooper_union_the_juilliard_school.wav | base64 | curl -s -H "Content-Type: application/octet-stream" --data @- 'localhost:5000/farePrediction' | jq


# GAE endpoint

export GAE_URL="https://ml-fare-prediction-347604.ue.r.appspot.com"
curl "$GAE_URL/textToSpeech?text=I+would+like+to+go+from+Chelsea+Market+to+The+Juilliard+School."
curl -X POST "$GAE_URL/speechToText"

curl "$GAE_URL/directions?origin=Brooklyn+Botanic+Garden&destination=Chelsea+Market"
curl "$GAE_URL/namedEntities?text=Flatiron+Building+and+Brooklyn+Bridge"

#curl -X POST "$GAE_URL/predict"
curl -d '[{"pickup_datetime":"2015-05-20 01:40:28 UTC","pickup_longitude":-73.9714126587,"pickup_latitude":40.7604141235,"dropoff_longitude":-73.9626083374,"dropoff_latitude":40.7623062134,"passenger_count":2}]' -H 'Content-Type: application/json' "$GAE_URL/predict"
#curl -X POST "$GAE_URL/farePrediction"
cat the_cooper_union_the_juilliard_school.wav | base64 | curl -s -H "Content-Type: application/octet-stream" --data @- "$GAE_URL/farePrediction" |  jq
