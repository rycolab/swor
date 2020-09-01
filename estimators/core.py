import utils
import io_utils
import numpy as np

class Estimator:
	"""
	Computations in log space!!
	"""

	def __init__(self, args):
		self._count = 0
		self._weight = self._total = utils.NEG_INF
		self.normalize = not args.no_normalization
		self.sign = 1

	def add_value(self, hypo, weight, **kwargs):
		raise NotImplementedError

	def increment(self, value, weight):
		log_value = np.log(abs(value)) if value else utils.NEG_INF
		sign = 1 if value >= 0 else -1
		self.sign, self._total = utils.signed_log_add(self._total, log_value + weight, self.sign, sign)
		self._weight = utils.log_add(self._weight, weight)
		self._count += 1

	def estimate(self):
		if self.normalize:
			return self.sign*np.exp(self._total - self._weight)
		return self.sign*np.exp(self._total)

	def reset(self):
		self._count = 0
		self._weight = self._total = utils.NEG_INF

	@staticmethod
	def add_args(parser):
		parser.add_argument("--importance_sampling", default=False, action="store_true",
			help="Use importance sampling techniques for building estimators")
		parser.add_argument("--no_normalization", default=False, action="store_true",
			help="Use importance sampling techniques for building estimators")


class BleuScoreEstimator(Estimator):

	
	name='bleu'
	def __init__(self, args):
		from mosestokenizer import MosesDetokenizer

		super(BleuScoreEstimator, self).__init__(args)
		trgt_language = args.trgt_language
		self.detokenizer = MosesDetokenizer(trgt_language) 

	def add_value(self, hypo, weight, ref=None):
		if not ref:
			return 0
		sen = io_utils.decode(hypo.trgt_sentence)
		value = utils.sentence_bleu(sen, ref, self.detokenizer)
		self.increment(value, weight)
		return value

	@staticmethod
	def add_args(parser):
		Estimator.add_args(parser)
		parser.add_argument("--trgt_language", default=None, type=str)


class ModelEntropyEstimator(Estimator):

	name='entropy'
	def __init__(self, args):
		super(ModelEntropyEstimator, self).__init__(args)

	def add_value(self, hypo, weight, **kwargs):
		value = -hypo.base_score if hypo.base_score else -hypo.total_score
		self.increment(value, weight)
		return value
