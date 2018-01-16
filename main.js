// create a neural net that will learn XOR with 2 input, 1x3 hidden and 1 output

//function defs
function sigmoid(t) {
    return 1/(1+Math.pow(Math.E, -t));
}

function node_init(in_nodes,hidden_nodes,out_nodes){
	var inputs = [];
		for (var i =0;i<in_nodes;i++){inputs.push(new inputNode(0));}
	var hiddens = [];
		for (var i =0;i<hidden_nodes;i++){hiddens.push(new hiddenNode());}
	var outputs = [];
		for (var i =0;i<out_nodes;i++){outputs.push(new outputNode());}
	hiddens.forEach(function(item,index){
		item.initWeights(inputs)
	})
	outputs.forEach(function(item,index){
		item.initWeights(hiddens)
	})
	return {'i':inputs,'h':hiddens,'o':outputs}
}

function propogate(nodes, input){
	nodes.i.forEach(function(item,index){
		item.value=input[index]
	})
	nodes.h.forEach(function(item,index){
		item.propogate(nodes.i)
	})
	nodes.o.forEach(function(item,index){
		item.propogate(nodes.h)
	})
}

function backPropogate(learning_rate,nodes,target){
	nodes.o.forEach(function(item,index){
		item.backPropogate(learning_rate,target[index],nodes.h)
	})
	nodes.h.forEach(function(item,index){
		item.backPropogate(learning_rate,nodes.i,nodes.o,index)
	})
	nodes.o.forEach(function(item,index){
		item.confirmBackPropogation()
	})
	nodes.h.forEach(function(item,index){
		item.confirmBackPropogation()
	})
}

function test(nodes,input){
	propogate(nodes,input)
	var out = {}
	nodes.o.forEach(function(item,index){
		out[index]=item.value
	})
	console.log([input,out])
}

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min)) + min; //The maximum is exclusive and the minimum is inclusive
}

//class defs
class inputNode{
	constructor(startValue){
		this._value=startValue;
	}
	get value(){
		return this._value;
	}
	
	set value(value){
		this._value=value;
	}
}

class hiddenNode{
	constructor(){
		this.bias = 0.0;
		this.new_weights={};
		this.weights={};
		this._net=0;
		this._output=0.0;
		this._delta=0;
	}
	
	get value(){
		return this._output;
	}	
		
	setWeight(index,value){
		this.weights[index]=value;
	}
	
	initWeights(inputs){
		var _this=this
		inputs.forEach(function(item,index){
			_this.setWeight(index,Math.random())
		})
	}
	
	propogate(inputs){
		var _this=this;
		_this._net=_this.bias;
		inputs.forEach(function(item,index){
			_this._net=_this._net+item.value*_this.weights[index]
		})
		_this._output=sigmoid(_this._net);
	}
	
	backPropogate(learning_rate,inputs,outputs,h_index){
		var _this=this;
		_this._delta=0;
		outputs.forEach(function(item,o_index){
			_this._delta=_this._delta+item.delta*item.weights[h_index]
		})
		_this._delta=_this._delta*_this._output*(1-_this._output)
		inputs.forEach(function(item,i_index){
			_this.new_weights[i_index]=_this.weights[i_index]-learning_rate*_this._delta*item.value
		})
	}
	
	confirmBackPropogation(){
		this.weights=this.new_weights
	}
}

class outputNode{
	constructor(){
		this.bias = 0.0;
		this.new_weights={};
		this.weights={};
		this._delta;
		this._net=0;
		this._output=0.0;
		
	}
	
	get delta(){
		return this._delta
	}
	
	get value(){
		return this._output;
	}	
		
	setWeight(index,value){
		this.weights[index]=value;
	}
	
	initWeights(hiddens){
		var _this=this
		hiddens.forEach(function(item,index){
			_this.setWeight(index,((Math.random()*4)-2))
		})
	}
	
	propogate(hiddens){
		var _this=this;
		_this._net=_this.bias;
		hiddens.forEach(function(item,index){
			_this._net=_this._net+item.value*_this.weights[index]
		})
		_this._output=sigmoid(_this._net);
	}
	
	backPropogate(learning_rate,target,hiddens){
		var _this=this
		_this._delta=-(target-_this._output)*_this._output*(1-_this._output)
		hiddens.forEach(function(item,index){
			_this.new_weights[index]=_this.weights[index]-learning_rate*_this.delta*item.value
		})
	}
	
	confirmBackPropogation(){
		this.weights=this.new_weights
	}
}

function main(){
	var L_R=0.1;
	var nodes=node_init(2,3,1)
	for(i=0;i<1e5;i++){
		
		var a = Math.random();
		var b = Math.random();
		var c= a>b ? 1 : 0;
		
		propogate(nodes,[a,b])
		backPropogate(L_R,nodes,[c])
	}
	return nodes
}