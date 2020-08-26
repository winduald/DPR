def mrr(out, th):
  """Computes MRR.

  Args:
    out: dict where each key maps to a ranked list of candidates. Each values
    is "true" or "false" indicating if the candidate is relevant or not.
  """
  n = len(out)
  MRR = 0.0
  for qid in out:
    candidates = out[qid]
    for i in xrange(min(th, len(candidates))):
      if candidates[i] == "true":
        MRR += 1.0 / (i + 1)
        break
  return MRR * 100.0 / n


def precision(out, th):
  precisions = [0.0]*th
  n = 0
  for qid in out:
    candidates = out[qid]
    if all(x == "false" for x in candidates):
      continue
    for i in xrange(min(th, len(candidates))):
      if candidates[i] == "true":
        precisions[i] += 1.0
        break
    n += 1
  for i in xrange(1, th):
    precisions[i] += precisions[i-1]  

  return [p*100/n for p in precisions]


def recall_of_1(out, th):
  precisions = [0.0]*th
  for qid in out:
    candidates = out[qid]
    if all(x == "false" for x in candidates):
      continue
    for i in xrange(min(th, len(candidates))):
      if candidates[i] == "true":
        precisions[i] += 1.0
        break
  for i in xrange(1, th):
    precisions[i] += precisions[i-1]  

  return [p*100/len(out) for p in precisions]


def cal_map(out, th):
  '''
  out: input is a dictionary
  out[query_id] = [(label, ...., )]. The first element needs to be label: "true" or "false". 
  Other elements can be anything
  '''
  num_queries = len(out)
  MAP = 0.0
  for qid in out:
    candidates = out[qid]
    # compute the number of relevant docs
    # get a list of precisions in the range(0,th)
    avg_prec = 0
    precisions = []
    num_correct = 0
    for i in range(min(th, len(candidates))):
      if candidates[i][0] == "true":
        num_correct += 1
        precisions.append(num_correct/(i+1))
    
    if precisions:
      avg_prec = sum(precisions)/len(precisions)
    
    MAP += avg_prec
  return MAP / num_queries