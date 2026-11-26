The model got the real record below wrong. Its true label is "{true_label}", but the model decided "{decision}"; the model's probability for its decision was {prob}.

The record, as a JSON object of path to value:
{seed}

Write {n} new records that the model would get wrong in the same way. Each keeps the true label "{true_label}" and keeps what made this record hard for the model, while it differs from this record and from the other new records in the values that matter: no near-copies. For each new record, change between 3 and 10 fields.{feedback}

Answer with {{"records": [{{"changes": {{"<path>": <new value>}}}}]}}, holding exactly {n} records.
