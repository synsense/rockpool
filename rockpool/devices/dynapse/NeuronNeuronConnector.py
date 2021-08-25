import CtxDynapse


class DynapseConnector:
    """
    Connector for DYNAP-se chip
    """

    def __init__(self):
        """
        Initialize the class with empty connections
        sending_connections_to: map of (key, value), where key is the neuron sending connection and value is a list of all connections that start with the key neuron
        receiving_connections_from: map of (key, value), where key is the neuron receiving connection and value is a list of all connections that end with the key neuron
        """

        self.sending_connections_to = {}
        self.receiving_connections_from = {}
        self.sending_virtual_connections_to = {}
        self.receiving_virtual_connections_from = {}

    def _save_connection(self, pre_neuron, post_neuron):
        self._add_to_list(self.sending_connections_to, pre_neuron, post_neuron)
        self._add_to_list(self.receiving_connections_from, post_neuron, pre_neuron)

    def _save_connection(self, pre_neuron, post_neuron):
        self._add_to_list(self.sending_connections_to, pre_neuron, post_neuron)
        self._add_to_list(self.receiving_connections_from, post_neuron, pre_neuron)

    def _save_virtual_connection(self, pre_neuron, post_neuron):
        self._add_to_list(self.sending_virtual_connections_to, pre_neuron, post_neuron)
        self._add_to_list(
            self.receiving_virtual_connections_from, post_neuron, pre_neuron
        )

    def _add_to_list(self, list_name, key, value):
        if key in list_name:
            list_name[key].append(value)
        else:
            list_name[key] = [value]

    def _remove_connection(self, pre_neuron, post_neuron):
        self._remove_from_list(self.sending_connections_to, pre_neuron, post_neuron)
        self._remove_from_list(self.receiving_connections_from, post_neuron, pre_neuron)

    def _remove_virtual_connection(self, pre_neuron, post_neuron):
        self._remove_from_list(
            self.sending_virtual_connections_to, pre_neuron, post_neuron
        )
        self._remove_from_list(
            self.receiving_virtual_connections_from, post_neuron, pre_neuron
        )

    def _remove_from_list(self, list_name, key, value):
        if key in list_name:
            try:
                list_name[key].remove(value)
                if not list_name[key]:
                    list_name.pop(key)
            except (ValueError, KeyError):
                print(
                    "The neuron {} has no connection with neuron {}".format(key, value)
                )
        else:
            raise IndexError("The neuron {} has no connections to remove".format(key))

    def add_connection(self, pre, post, synapse_type):
        """
        Connect two neurons.
        Attributes:
            pre: neuron that sends the connection
            post: neuron that receives the connection
            synapse_type: one of the four expected synapse types
        """

        # check if one of the neurons is virtual
        pre_virtual = pre.is_virtual()
        post_virtual = post.is_virtual()

        # if post_neuron is virtual, raise an error.
        # Virtual neurons cannot receive connections
        if post_virtual:
            raise Exception(
                "post neuron cannot be virtual. Virtual neurons do not receive connections."
            )

        if pre_virtual:
            self.add_virtual_connection(pre, post, synapse_type)
            return

        pre_srams = pre.get_srams()
        pre_core_id = pre.get_core_id()
        pre_chip_id = pre.get_chip_id()
        pre_neuron_id = pre.get_neuron_id()

        post_cams = post.get_cams()
        post_core_id = post.get_core_id()
        post_chip_id = post.get_chip_id()
        post_neuron_id = post.get_neuron_id()

        target_chip = post_chip_id

        # check if the pre can send a connection to target chip
        pre_available = False
        pre_sram = 0
        for i in range(len(pre_srams)):
            if pre_srams[i].get_target_chip_id() == target_chip:
                pre_available = True
                pre_sram = i
                break

        if not pre_available:
            for i in range(len(pre_srams)):
                if not pre_srams[i].is_used():
                    pre_available = True
                    pre_sram = i
                    break

        if not pre_available:
            raise Exception("pre neuron has no available outputs")

        # check if the post can receive a connection
        post_available = False
        for i in range(len(post_cams)):

            # if there is space left on post_cams
            if (
                post_cams[i].get_pre_neuron_id()
                + post_cams[i].get_pre_neuron_core_id() * 256
            ) == 0:
                post_available = True
                post_cam = i
                break

        if not post_available:
            raise Exception("post neuron has no available inputs")

        # connect
        pre_neuron_address = pre_core_id * 256 + pre_neuron_id
        post_neuron_address = post_core_id * 256 + post_neuron_id
        virtual_core_id = 0
        if pre_srams[pre_sram].is_used():
            virtual_core_id = pre_srams[pre_sram].get_virtual_core_id()
        else:
            virtual_core_id = pre_core_id

        core_mask = pre_srams[pre_sram].get_core_mask() | (1 << post_core_id)

        d = (post_chip_id & 1) - (pre_chip_id & 1)
        if d < 0:
            sx = True
        else:
            sx = False

        dx = abs(d)

        d = ((post_chip_id & 2) >> 1) - ((pre_chip_id & 2) >> 1)

        if d < 0:
            sy = False
        else:
            sy = True

        dy = abs(d)

        pre_srams[pre_sram].set_virtual_core_id(virtual_core_id)
        pre_srams[pre_sram].set_target_chip_id(post_chip_id)
        pre_srams[pre_sram].set_sx(sx)
        pre_srams[pre_sram].set_sy(sy)
        pre_srams[pre_sram].set_dx(dx)
        pre_srams[pre_sram].set_dy(dy)
        pre_srams[pre_sram].set_used(True)
        pre_srams[pre_sram].set_core_mask(core_mask)

        post_cams[post_cam].set_pre_neuron_id(pre_neuron_id)
        post_cams[post_cam].set_pre_neuron_core_id(pre_core_id)
        post_cams[post_cam].set_type(synapse_type)

        # CtxDynapse.dynapse.set_config_chip_id(pre_chip_id)
        # CtxDynapse.dynapse.write_sram(
        #    pre_neuron_address, pre_sram, virtual_core_id, sx, dx, sy, dy, core_mask
        # )
        #
        # if pre_chip_id != post_chip_id:
        #    CtxDynapse.dynapse.set_config_chip_id(post_chip_id)

        # CtxDynapse.dynapse.write_cam(
        #    pre_neuron_address, post_neuron_address, post_cam, synapse_type
        # )

        self._save_connection(pre, post)

    def add_connection_from_list(
        self, pre_neurons_list, post_neuron_list, synapse_types
    ):
        """
        Connect neurons using a python list of pre, post and synapse types.
        Attributes:
            pre_neurons_list: list with neurons that send the connection
            post_neuron_list: list with neurons that receive the connection
            synapse_types: list with the connection type between the neurons. It can be a list with one single
            element indicating all the connections are of the same type, otherwise, the size of the synapse types list must
            match the size of pre and post neurons list.
        """

        if len(pre_neurons_list) != len(post_neuron_list):
            print(
                "The number of pre and post neurons must be the same. No connection will be created."
            )
            return

        same_synapse_type = False
        if len(synapse_types) == 1:
            same_synapse_type = True

        if (len(pre_neurons_list) != len(synapse_types)) and (not same_synapse_type):
            print(
                "The number of synapses type must match the number of connections. No connection will be created."
            )
            return

        for i in range(len(pre_neurons_list)):
            self.add_connection(
                pre_neurons_list[i],
                post_neuron_list[i],
                synapse_types[0] if same_synapse_type else synapse_types[i],
            )

    def add_connection_from_file(self, connection_file_path):
        """
        Connects neurons reading the values from a file. The file must contain three elements per line: pre neuron, post neuron and synapse type.
        Attributes:
            connection_file: file that contains three elements in each row: pre neuron, post neuron and synapse type.
        """
        number_of_connections = 0

        with open(connection_file_path, "r") as fp:
            for i in fp.readlines():
                tmp = i.split(" ")

                # verify if there is 3 elements in the line
                if len(tmp) == 3:
                    # connect
                    self.add_connection(tmp[0], tmp[1], tmp[2])
                    number_of_connections += 1
                else:
                    print(
                        "Bad format error. Error in the line {}. The connections before this point were created.".format(
                            number_of_connections + 1
                        )
                    )

    def add_virtual_connection(self, pre, post, synapse_type):
        """
        Connect a virtual neuron with a real (on chip) neuron.
        Attributes:
            pre: neuron that sends the connection, it must be virtual
            post: neuron that receives the connection, it must not be virtual
            synapse_type: one of the four expected synapse types
        """

        if not pre.is_virtual():
            raise Exception("pre neuron must be virtual")

        if post.is_virtual():
            raise Exception("post neuron must not be virtual")

        pre_core_id = pre.get_core_id()
        pre_chip_id = pre.get_chip_id()
        pre_neuron_id = pre.get_neuron_id()

        post_cams = post.get_cams()
        post_core_id = post.get_core_id()
        post_chip_id = post.get_chip_id()
        post_neuron_id = post.get_neuron_id()

        # check if the post can receive a connection
        post_available = False
        for i in range(len(post_cams)):
            # if there is space left on post_cams
            if (
                post_cams[i].get_pre_neuron_id()
                + post_cams[i].get_pre_neuron_core_id() * 256
            ) == 0:
                post_available = True
                post_cam = i
                break

        if not post_available:
            raise Exception("post neuron has no available inputs")

        # connect
        pre_neuron_address = pre_core_id * 256 + pre_neuron_id
        post_neuron_address = post_core_id * 256 + post_neuron_id
        virtual_core_id = pre_core_id

        post_cams[post_cam].set_pre_neuron_id(pre_neuron_id)
        post_cams[post_cam].set_pre_neuron_core_id(pre_core_id)
        post_cams[post_cam].set_type(synapse_type)

        # CtxDynapse.dynapse.set_config_chip_id(post_chip_id)
        # CtxDynapse.dynapse.write_cam(
        #    pre_neuron_address, post_neuron_address, post_cam, synapse_type
        # )

        self._save_virtual_connection(pre, post)

    def remove_connection(self, pre_neuron, post_neuron):
        """
        Delete the connection between two neurons.
        Attributes:
            pre_neuron: neuron that sends the connection
            post_neuron: neuron that receives the connection
        """

        # check if one of the neurons is virtual
        pre_virtual = pre_neuron.is_virtual()
        post_virtual = post_neuron.is_virtual()

        # if post_neuron is virtual, raise an error.
        # Virtual neurons do not receive connections, thus there is no connection to remove
        if post_virtual:
            raise Exception("post neuron is virtual, there is no connection to remove.")

        if pre_virtual:
            self.remove_virtual_connection(pre_neuron, post_neuron)
            return

        # first, try to remove the neurons from the lists. This will raise an exception if the neurons aren't connected.
        # todo: handle exception
        self._remove_connection(pre_neuron, post_neuron)

        # now, we can remove the connections on chip

        # get info about pre and post neurons
        pre_srams = pre_neuron.get_srams()
        pre_core_id = pre_neuron.get_core_id()
        pre_chip_id = pre_neuron.get_chip_id()
        pre_neuron_id = pre_neuron.get_neuron_id()

        post_cams = post_neuron.get_cams()
        post_core_id = post_neuron.get_core_id()
        post_chip_id = post_neuron.get_chip_id()
        post_neuron_id = post_neuron.get_neuron_id()

        pre_neuron_address = pre_core_id * 256 + pre_neuron_id
        post_neuron_address = post_core_id * 256 + post_neuron_id

        # check what sram sends a connection to post neuron
        pre_sram = 0
        for i in range(len(pre_srams)):
            if pre_srams[i].get_target_chip_id() == post_chip_id:
                pre_sram = i
                break

        pre_virtual_core_id = pre_srams[pre_sram].get_virtual_core_id()

        # check what cam receives a connection from pre neuron
        post_cam = 0
        for i in range(len(post_cams)):
            if post_cams[i] == pre_neuron_address:
                post_cam = i
                break

        # CtxDynapse.dynapse.set_config_chip_id(post_chip_id)

        ## information of post-synaptic neuron, setting the address of pre-synaptic neuron to zero
        # CtxDynapse.dynapse.write_cam(0, post_neuron_address, post_cam, 0)
        post_cams[post_cam].set_pre_neuron_id(0)
        post_cams[post_cam].set_pre_neuron_core_id(0)

        ## updating pre-synaptic neuron
        # if pre_chip_id != post_chip_id:
        #    CtxDynapse.dynapse.set_config_chip_id(pre_chip_id)

        # if there is no other connections from pre neuron, set it to zero and mark it as unused
        if pre_neuron not in self.sending_connections_to:
            # information of pre-synaptic neuron, setting the address of post-synaptic neuron to zero
            # CtxDynapse.dynapse.write_sram(
            #    pre_neuron_address, pre_sram, 0, 0, 0, 0, 0, 0
            # )
            pre_srams[pre_sram].set_used(False)
            pre_srams[pre_sram].set_virtual_core_id(0)
            pre_srams[pre_sram].set_target_chip_id(0)
            pre_srams[pre_sram].set_sx(0)
            pre_srams[pre_sram].set_sy(0)
            pre_srams[pre_sram].set_dx(0)
            pre_srams[pre_sram].set_dy(0)
            pre_srams[pre_sram].set_core_mask(0)

        # if there are other connections, check if they are projecting to the same core as the post-neuron
        else:
            post_list = self.sending_connections_to[pre_neuron]
            found_post_same_core = False
            for element in post_list:
                if element.get_core_id() == post_core_id:
                    found_post_same_core = True

            # if none of the connection go to the same core as the post_neuron, we set the corresponding bit of the core mask to 0
            if not found_post_same_core:
                core_mask = pre_srams[pre_sram].get_core_mask() & ~(0 << post_core_id)
                # CtxDynapse.dynapse.write_sram(
                #    pre_neuron_address, pre_sram, 0, 0, 0, 0, 0, core_mask
                # )
                pre_srams[pre_sram].set_core_mask(core_mask)

    def remove_connection_from_list(self, pre_neurons_list, post_neuron_list):
        """
        Delete the connection between two list of neurons. The number of elements in each list must be the same.
        Attributes:
            pre_neurons_list: list of neurons that send the connection
            post_neurons_list: list of neuron that receive the connection
        """
        if len(pre_neurons_list) != len(post_neuron_list):
            print(
                "The number of pre and post neurons must be the same. No connection was removed."
            )
            return

        for i in range(len(pre_neurons_list)):
            # todo: handle exception
            self.remove_connection(pre_neurons_list[i], post_neuron_list[i])

    def remove_connection_from_file(self, unconnect_file_path):
        """
        Delete the connection between neurons reading the values from a file. The file must contain two elements per line: pre neuron and post neuron.
        Attributes:
            connection_file: file that contains two elements in each row: pre neuron and post neuron.
        """
        number_of_connections_removed = 0

        with open(unconnect_file_path, "r") as fp:
            for i in fp.readlines():
                tmp = i.split(" ")

                # verify if there is 2 elements in the line
                if len(tmp) == 2:
                    # unconnect
                    # todo: catch exception
                    self.remove_connection(tmp[0], tmp[1])
                    number_of_connections_removed += 1
                else:
                    print(
                        "Bad format error. Error in the line {}. The connections before this point were removed.".format(
                            number_of_connections_removed + 1
                        )
                    )

    def remove_virtual_connection(self, pre_neuron, post_neuron):
        """
        Delete the connection between a virtual neuron and a real (on chip) neuron.
        Attributes:
            pre_neuron: neuron that sends the connection - must be virtual
            post_neuron: neuron that receives the connection - must not be virtual
        """

        if not pre_neuron.is_virtual():
            raise Exception("pre neuron must be virtual")

        if post_neuron.is_virtual():
            raise Exception("post neuron must not be virtual")

        # check if this connection is on the lists.
        # This will raise an exception if the neurons aren't connected.
        # todo: handle exception
        self._remove_virtual_connection(pre_neuron, post_neuron)

        # now, we can remove the connection on chip
        # we just need to clean the cam of the post, pre neuron is virtual

        # get info about pre and post neurons
        pre_core_id = pre_neuron.get_core_id()
        pre_neuron_id = pre_neuron.get_neuron_id()

        post_cams = post_neuron.get_cams()
        post_core_id = post_neuron.get_core_id()
        post_chip_id = post_neuron.get_chip_id()
        post_neuron_id = post_neuron.get_neuron_id()

        pre_neuron_address = pre_core_id * 256 + pre_neuron_id
        post_neuron_address = post_core_id * 256 + post_neuron_id

        # check what cam receives a connection from pre neuron
        post_cam = 0
        for i in range(len(post_cams)):
            if post_cams[i] == pre_neuron_address:
                post_cam = i
                break

        # CtxDynapse.dynapse.set_config_chip_id(post_chip_id)
        # information of post-synaptic neuron, setting the address of pre-synaptic neuron to zero
        # CtxDynapse.dynapse.write_cam(0, post_neuron_address, post_cam, 0)
        post_cams[post_cam].set_pre_neuron_id(0)
        post_cams[post_cam].set_pre_neuron_core_id(0)

    def remove_sending_connections(self, neuron):
        """
        Remove all connections leaving the informed neuron
        Attributes:
            neuron: the neuron passed as parameter will be considered the pre-synaptic neuron, and all the connections that leave this neuron will be removed.
        """

        # todo: handle exception
        if neuron in self.sending_connections_to:
            connections = self.sending_connections_to[neuron]
            for i in connections:
                self.remove_connection(neuron, i)

    def remove_receiving_connections(self, neuron):
        """
        Remove all connections arriving in the informed neuron
        Attributes:
            neuron: the neuron passed as parameter will be considered the post-synaptic neuron, and all the connections that are sent to this neuron will be removed.
        """
        if neuron.is_virtual:
            raise Exception(
                "neuron {} is virtual and receives no connection".format(neuron)
            )

        # todo: handle exception
        if neuron in self.receiving_connections_from:
            connections = self.receiving_connections_from[neuron]
            for i in connections:
                self.remove_connection(i, neuron)

    def remove_all_connections(self, neuron):
        """
        Remove all connections of a neuron, i.e., all the connections that the neuron send and receive will be removed.
        Attributes:
            neuron: the neuron that will have all its connections removed.
        """
        self.remove_sending_connections(neuron)

        if not neuron.is_virtual:
            self.remove_receiving_connections(neuron)


if __name__ == "__main__":

    model = CtxDynapse.model
    neurons = model.get_shadow_state_neurons()
    dynapse_connector = DynapseConnector()

    if len(neurons) > 2:
        dynapse_connector.add_connection(neurons[0], neurons[1], 3)
        dynapse_connector.add_connection(neurons[0], neurons[2], 3)
    else:
        print("missing neurons to connect")
