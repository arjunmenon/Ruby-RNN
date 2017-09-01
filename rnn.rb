require 'numo/narray'
require 'pp'

def sigmoid(x)
	output = 1/(1 + Numo::NMath.exp(-x))
end

def sigmoid_output_to_derivative(output)
	output * (1 - output)
end

int2binary = {}
binary_dim = 8

largest_number = 2 ** binary_dim

i = Numo::UInt8[0...largest_number]

bin = Numo::Bit.from_binary(i.to_binary,[i.size,8]).reverse(1)

largest_number.times { |o| int2binary[o] = bin.to_a[o] }

alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

synapse_0 = 2*Numo::DFloat.new(input_dim,hidden_dim).rand - 1

synapse_1 = 2*Numo::DFloat.new(hidden_dim,output_dim).rand - 1

synapse_h = 2*Numo::DFloat.new(hidden_dim,hidden_dim).rand - 1

synapse_0_update = synapse_0.new_zeros

synapse_1_update = synapse_1.new_zeros

synapse_h_update = synapse_h.new_zeros

10000.times do |j|
	a_int =  Numo::Int32.new().rand(largest_number/2).to_i
	a = int2binary[a_int]

	b_int =  Numo::Int32.new().rand(largest_number/2).to_i
	b = int2binary[b_int]

	c_int = a_int + b_int
	c = int2binary[c_int]

	d = Numo::NArray[c].new_zeros.to_a

	overallError = 0


    layer_2_deltas = []
    layer_1_values = []
    layer_1_values.push(Numo::DFloat.zeros(hidden_dim))

    binary_dim.times do |position|

    	# generate input and output
        x_input = Numo::NArray[[a[binary_dim - position - 1],b[binary_dim - position - 1]]]
        y = Numo::NArray[[c[binary_dim - position - 1]]].transpose

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid((x_input.dot synapse_0) + (layer_1_values[-1].dot synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(layer_1.dot synapse_1)

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.push((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += layer_2_error[0].abs

        # decode estimate so we can print it out
        # puts layer_2.inspect
        # puts layer_2[0].round
        d[binary_dim - position - 1] = layer_2[0].round
        
        # store hidden layer so we can use it in the next timestep
        # layer_1_values.append(copy.deepcopy(layer_1))
        layer_1_values.push(Marshal.load(Marshal.dump(layer_1))) # not efficient. it's just dirty hack!
    end

    future_layer_1_delta = Numo::DFloat.zeros(hidden_dim)

    binary_dim.times do |position|

    	x_input = Numo::NArray[[a[position],b[position]]]
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]

        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer

        layer_1_delta = (future_layer_1_delta.dot(synapse_h.transpose) + layer_2_delta.dot(synapse_1.transpose)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        
        if layer_1.shape.size < 2
        	layer_1 = layer_1.expand_dims(1)
        end
        # synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_1_update += (layer_1).transpose.dot(layer_2_delta)
        
        if prev_layer_1.shape.size < 2
        	prev_layer_1 = prev_layer_1.expand_dims(1)
        end
        # synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_h_update += (prev_layer_1).transpose.dot(layer_1_delta)
        
        synapse_0_update += x_input.transpose.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    end

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # print out progress
    if(j % 1000 == 0)
        pp "Error:" + (overallError.inspect)
        pp "Pred:" + (d.inspect)
        pp "True:" + (c.inspect)
        out = 0
        # for index,x in enumerate(reversed(d)):
            # out += x*(2**index)
        d.reverse.each_with_index do |x, index|
        	out += x*(2**index)
        end
        pp (a_int.inspect) + " + " + (b_int.inspect) + " = " + (out.inspect)
        pp "------------"
        pp d.inspect
        # pp d.each
    end
        
end
