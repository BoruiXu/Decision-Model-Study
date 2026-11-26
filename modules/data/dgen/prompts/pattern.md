The model gets a group of records wrong. What the records of this group have in common:
{pattern}

How to write new records of this group:
{strategy}

A real record of the group, whose true label is "{true_label}", as a JSON object of path to value:
{seed}

Following the pattern and the strategy, write {n} new records of this group. Each starts from this record and keeps its true label "{true_label}", and differs from it and from the other new records in the values that matter: no near-copies. For each new record, change between 3 and 10 fields.{feedback}

Answer with {{"records": [{{"changes": {{"<path>": <new value>}}}}]}}, holding exactly {n} records.
