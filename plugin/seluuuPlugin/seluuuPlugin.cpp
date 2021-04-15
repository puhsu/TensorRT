/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "seluuuPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>

using namespace nvinfer1;
const int NUM_COORDCONV_CHANNELS = 2;

namespace
{
const char* SELUUU_PLUGIN_VERSION{"1"};
const char* SELUUU_PLUGIN_NAME{"Seluuu"};
} // namespace

PluginFieldCollection SeluuuPluginCreator::mFC{};
std::vector<PluginField> SeluuuPluginCreator::mPluginAttributes;

SeluuuPlugin::SeluuuPlugin() {}

SeluuuPlugin::SeluuuPlugin(int dataDim_) {
    dataDim = dataDim_;
}

SeluuuPlugin::SeluuuPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    dataDim = read<int>(d);
    ASSERT(d == a + length);
}

int SeluuuPlugin::getNbOutputs() const
{
    return 1;
}

int SeluuuPlugin::initialize()
{
    return STATUS_SUCCESS;
}

void SeluuuPlugin::terminate() {}

Dims SeluuuPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    return inputs[0];
}

size_t SeluuuPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

size_t SeluuuPlugin::getSerializationSize() const
{
    return sizeof(int);
}

void SeluuuPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, dataDim);
    ASSERT(d == a + getSerializationSize());
}

void SeluuuPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize)
{
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);

    dataDim = 1;
    for (int i = 0; i < inputDims->nbDims; ++i) {
        dataDim *= inputDims->d[i];
    }
}

bool SeluuuPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char* SeluuuPlugin::getPluginType() const
{
    return SELUUU_PLUGIN_NAME;
}

const char* SeluuuPlugin::getPluginVersion() const
{
    return SELUUU_PLUGIN_VERSION;
}

void SeluuuPlugin::destroy()
{
    delete this;
}

IPluginV2Ext* SeluuuPlugin::clone() const
{
    auto* plugin = new SeluuuPlugin(dataDim);
    return plugin;
}

void SeluuuPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* SeluuuPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType SeluuuPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

bool SeluuuPlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool SeluuuPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Plugin creator
SeluuuPluginCreator::SeluuuPluginCreator() {}

const char* SeluuuPluginCreator::getPluginName() const
{
    return SELUUU_PLUGIN_NAME;
}

const char* SeluuuPluginCreator::getPluginVersion() const
{
    return SELUUU_PLUGIN_VERSION;
}

const PluginFieldCollection* SeluuuPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* SeluuuPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    SeluuuPlugin* plugin = new SeluuuPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* SeluuuPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    SeluuuPlugin* plugin = new SeluuuPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
