#!encoding=utf8

import math


## 모델파라미터가 주어지고, 데이터(feature vector)가 들어오면 예측하는 함수이다
## Logistic 함수를 사용한다
def predict(model_feature_weight_dict, feature_vector_dict):
	## 예측값을 0.0 으로 초기화한다음
	sum_wx = 0.0
	## 각각의 피쳐값에 각각의 피쳐 가중치를 곱한다음 다 더한다
	for feature_name, value in feature_vector_dict.iteritems():
		## 피쳐벡터에는 원래 피쳐값만 있고 맞춰야 되는 click 여부는 없는게 맞는것이겠지만...
		## 코딩 편의상 click 여부를 피쳐벡터에 넣어 놓았기 때문에 피쳐네임이 'click'인 경우는 무시한다
		if feature_name == 'click': continue

		sum_wx += model_feature_weight_dict[feature_name]

	## 로지스틱 함수에 넣어서 예측값으로 리턴한다 
	return 1.0 / (1.0 + math.pow(math.e, -1.0 * sum_wx))


## 새로운 데이터가 들어오면, 예측해본 값과 실제값을 비교해서 모델 파라미터를 튜닝해나간다
def update_weight(model_feature_weight_dict, feature_vector_dict):
	## eta 값은 편의상 모든 피쳐에 대해서 동일하게 부여하자
	eta = 1.0e-2
	for feature_name, value in feature_vector_dict.iteritems():
		## 피쳐벡터에는 원래 피쳐값만 있고 맞춰야 되는 click 여부는 없는게 맞는것이겠지만...
		## 코딩 편의상 click 여부를 피쳐벡터에 넣어 놓았기 때문에 피쳐네임이 'click'인 경우는 무시한다
		if feature_name == 'click': continue

		## 예측값을 가지고 온 다음에
		f = predict(model_feature_weight_dict, feature_vector_dict)
		## 실제값을 가지고 와서
		click = feature_vector_dict['click']
		## 예측값과 실제값의 차이로 모델 파라미터의 변경분을 (LogLoss의 미분을 참조하여) 구한다음
		gradient = (f - click) * feature_vector_dict[feature_name] 
		## 모델 파라미터를 업데이트한다
		weight = model_feature_weight_dict[feature_name]
		new_weight = weight - eta * gradient
		model_feature_weight_dict[feature_name] = new_weight

	## print model_feature_weight_dict



def main():
	## 모델은 Logistic Regression 으로 정하고, model_feature_weight_dict 에 모델의 파라미터들이 들어간다
	## 예를 들어 device_type:0, device_type:1 등이 정해져야 되는 모델 파라미터들이며, 구체적으로는 'device_type:0':0.5 식으로 저장된다
	## 'device_type:0':0.5 의 의미는 다음과 같다. device_type:0 이라는 모델 파라미터의 값은 0.5 이다는 뜻
	## y_intercept 라는 모델 파라미터는 처음에 시작할때, 초기값을 1.0 으로 세팅하고, 이후에 데이터가 들어오면서 튜닝해나가도록 하자
	model_feature_weight_dict = {'y_intercept':1.0}



	## train_subset.csv 라는 파일로 모델을 학습하자
	fin = open('train_subset.csv', 'r')


	## 전체 CTR을 구하기 위해서 노출, 클릭값을 저장할 변수이다
	imp = 0.0
	clk = 0.0

	## logistic regression 을 통한 예측시, 나오는 에러값들을 저장할 변수이다
	logistic_regression_loss = 0.0
	## 전체 CTR 을 통한 예측시, 나오는 에러값들을 저장할 변수이다
	total_ctr_loss = 0.0


	## device_type별 CTR을 구하기 위해서 노출, 클릭값을 저장할 변수이다
	device_type_imp_clk_dict = {'device_type:0':[0.0, 0.0], 'device_type:1':[0.0, 0.0], 'device_type:4':[0.0, 0.0], 'device_type:5':[0.0, 0.0]}
	## device_type별 LogLoss을 구하기 위해서 노출, 클릭값을 저장할 변수이다
	device_type_loss_dict = {'device_type:0':[0.0, 0.0], 'device_type:1':[0.0, 0.0], 'device_type:4':[0.0, 0.0], 'device_type:5':[0.0, 0.0]}


	HEAD_LINE = True
	FEATURE_HEAD_LIST = []
	for line in fin:
		tokens	= line.rstrip().split(',')

		## 첫번째 라인은 피쳐 네임(예를 들어 'device_type')이 있고, 이 피쳐네임이 모델파라미터에 들어가야 되기 때문에(예를 들어 'device_type:0') 해당 내용을 기억하고 있어야 한다
		if HEAD_LINE:
			FEATURE_HEAD_LIST = tokens[:]
			HEAD_LINE = False
			continue
		
		## 데이터 하나는 피쳐 벡터로 나타낸다
		## 모델파라미터에 'y_intercept' 가 반드시 들어가게 되는데, 이에 해당하는 데이터 값을 1.0 으로 고정한다고 배운적이 있다
		## 이를 아래와 같이 표현하기로 하자
		feature_vector_dict = {'y_intercept':1.0}


		## 데이터 한줄을 읽어서 각각의 피쳐값을 추출하는게 먼저다
		for i in range(len(FEATURE_HEAD_LIST)):

			## 피쳐에서 'id', 'hour'은 일단 고려안하기로 하자
			if FEATURE_HEAD_LIST[i] in ('id', 'hour'): continue
			## 피쳐에서 'device_type', 'C1', 'C15', 'C16' 만을 일단 고려하기로 하고
			## 'click' 은 클릭 여부의 실제값이므로 이 값도 역시 고려하자
			if FEATURE_HEAD_LIST[i] not in ('click', 'device_type', 'C1', 'C15', 'C16'): continue

			## 'click'인 경우 피쳐벡터에는 'click':0 또는 'click':1 로 저장하기로 하자
			if FEATURE_HEAD_LIST[i] == 'click':
				feature_vector_dict[FEATURE_HEAD_LIST[i]] = int(tokens[i])
			## 그렇지 않고 'device_type', 'C1', 'C15', 'C16' 인 경우
			## 데이터에서 해당 피쳐값을 읽어서 '피쳐네임:피쳐값'으로 모델파라미터를 만들자
			## 예를 들어 device_type 에 해당하는 피쳐값이 0 인 경우, 'device_type:0'으로 만든다음
			## 이 값을 모델 파라미터로 세팅한다
			## 그런다음 모델 파라미터를 저장하고 있는 model_feature_weight_dict에는 
			## 'device_type:0':0.0 으로 초기화해서 0.0 을 시작으로 파라미터 값을 업데이트하기로 하자
			else:
				feature_name = '%s:%s' % (FEATURE_HEAD_LIST[i], tokens[i])
				feature_vector_dict[feature_name] = 1

				if feature_name not in model_feature_weight_dict:
					model_feature_weight_dict[feature_name] = 0.0

		## 데이터 한줄을 읽었으므로, 예측값과 실제클릭여부값을 가지고 온다
		prd_y = predict(model_feature_weight_dict, feature_vector_dict)
		tgt_y = feature_vector_dict['click']


		## 모든데이터의 CTR 을 저장하기 위한 간단한 코드이다
		if imp > 0.0:
			pctr = clk/imp

		## device_type별로 CTR을 저장하기 위한 간단한 코드이다
		device_type = filter(lambda x: x.find('device') >= 0, feature_vector_dict.keys())[0]
		if device_type_imp_clk_dict[device_type][0] > 0.0:
			device_type_pctr = device_type_imp_clk_dict[device_type][1] / device_type_imp_clk_dict[device_type][0]
			## print device_type, device_type_pctr
			## print device_type_imp_clk_dict[device_type][1], device_type_imp_clk_dict[device_type][0]


		## 데이터가 2만줄 이상인 경우에 에러들을 누적하고 중간 결과를 출력하자
		if imp > 20000.0:
			## 1. logistic regression 예측을 통해 나오는 에러를 누적한다 
			logistic_regression_loss 	+= tgt_y * (-1.0 * math.log(prd_y)) + (1.0 - tgt_y) * (-1.0 * math.log(1.0 - prd_y))
			## 2. 전체 CTR을 통한 예측을 통해 나오는 에러를 누적한다 
			total_ctr_loss 				+= tgt_y * (-1.0 * math.log(pctr)) + (1.0 - tgt_y) * (-1.0 * math.log(1.0 - pctr))

			## 3. device_type별 CTR을 통한 예측을 통해 나오는 에러를 누적한다
			device_type_loss_dict[device_type][0] += 1
			device_type_loss_dict[device_type][1] += tgt_y * (-1.0 * math.log(device_type_pctr)) + (1.0 - tgt_y) * (-1.0 * math.log(1.0 - device_type_pctr))

			## 1, 2, 3 을 출력해본다
			print '%s\t%s\t%s\t%s\tlr_loss:%s\tt_ctr:%s\tdevice_ctr:%s' % (int(imp), tgt_y, prd_y, device_type_pctr, logistic_regression_loss, total_ctr_loss, sum(map(lambda x: x[1], device_type_loss_dict.values())))
			## print map(lambda x: (x[0], x[1][1]/ x[1][0]), device_type_imp_clk_dict.items())

		## 모델 파라미터를 업데이트할 시기이다
		update_weight(model_feature_weight_dict, feature_vector_dict)

		## 노출, 클릭량을 업데이트한다
		imp += 1.0
		clk += tgt_y
		device_type_imp_clk_dict[device_type][0] += 1
		device_type_imp_clk_dict[device_type][1] += tgt_y

	## 마지막에 logistic regression 에 들어가는 모델 파라미터를 출력해본다
	result_list = model_feature_weight_dict.items()
	result_list.sort()
	print result_list


if __name__=='__main__':
	main()




