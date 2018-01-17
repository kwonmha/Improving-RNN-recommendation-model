from keras import optimizers

def update_manager_command_parser(parser):
	parser.add_argument('--u_m', dest='update_manager', choices=['adagrad', 'adadelta', 'rmsprop', 'adam'], help='Update mechanism',
						default='adagrad')
	parser.add_argument('--u_l', help='Learning rate, only recommended for adagrad', default=0.1, type=float)


def get_update_manager(args):
	if args.update_manager == 'adagrad':
		return Adagrad(learning_rate=args.u_l)
	elif args.update_manager == 'adadelta':
		return Adadelta()
	elif args.update_manager == 'rmsprop':
		return RMSProp()
	elif args.update_manager == 'adam':
		return Adam()
	else:
		raise ValueError('Unknown update option')


class Adagrad(object):
	def __init__(self, learning_rate=0.01, **kwargs):
		super(Adagrad, self).__init__(**kwargs)

		self.learning_rate = learning_rate
		self.name = 'Ug_lr' + str(self.learning_rate)

	def __call__(self):
		return optimizers.Adagrad(lr=self.learning_rate)


class Adadelta(object):
	def __init__(self, learning_rate=1.0, rho=0.95, **kwargs):
		super(Adadelta, self).__init__(**kwargs)

		self.learning_rate = learning_rate
		self.rho = rho
		self.name = 'Ud_lr' + str(self.learning_rate) + '_rho' + str(self.rho)

	def __call__(self):
		return optimizers.Adadelta()


class RMSProp(object):
	def __init__(self, learning_rate=0.001, rho=0.9, **kwargs):
		super(RMSProp, self).__init__(**kwargs)

		self.learning_rate = learning_rate
		self.rho = rho
		self.name = 'Ur_lr' + str(self.learning_rate) + '_rho' + str(self.rho)

	def __call__(self):
		return optimizers.RMSprop()

class Adam(object):
	def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, **kwargs):
		super(Adam, self).__init__(**kwargs)

		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.name = 'Ua_lr' + str(self.learning_rate) + '_b1' + str(self.beta1) + '_b2' + str(self.beta2)

	def __call__(self):
		return optimizers.Adam()