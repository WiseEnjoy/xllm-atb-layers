/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "acl/acl.h"
#include "operations/aclnn/utils/op_api_resolver.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "obfuscation_setup_operation.h"

namespace {
using AclnnObfuscationSetupGetWorkspaceSizeFunc = aclnnStatus (*)(
    int32_t fdToClose, int32_t dataType, int32_t hiddenSize, int32_t tpRank,
    int32_t modelObfSeedId, int32_t dataObfSeedId, int32_t cmd, int32_t threadNum,
    aclTensor *fd, uint64_t *workspaceSize, aclOpExecutor **executor);
using AclnnObfuscationSetupFunc = aclnnStatus (*)(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);
} // namespace



namespace atb_speed {
namespace common {

ObfuscationSetupOperation::ObfuscationSetupOperation(const std::string &name,
    ObfuscationSetupParam param) : AclNNOperation(name), param_(param) {}

ObfuscationSetupOperation:: ~ObfuscationSetupOperation()
{
    ATB_SPEED_LOG_DEBUG("ObfuscationSetupOperation deconstructor");
    this->DestroyOperation();
}

atb::Status ObfuscationSetupOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG("ObfuscationSetupOperation infer shape start");
    if (inTensorDescs.size() != 0) {
        ATB_SPEED_LOG_ERROR("ObfuscationSetupOperation intensors should be 0, but get " <<
           inTensorDescs.size());
        return atb::ERROR_INVALID_TENSOR_SIZE;
    }
    outTensorDescs.at(DIM0).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(DIM0).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(DIM0).shape.dimNum = NUM1;
    outTensorDescs.at(DIM0).shape.dims[DIM0] = NUM1;
    ATB_SPEED_LOG_DEBUG("ObfuscationSetupOperation infer shape end");
    return 0;
}

uint32_t ObfuscationSetupOperation::GetInputNum() const { return DIM0; }

uint32_t ObfuscationSetupOperation::GetOutputNum() const { return NUM1; }

int ObfuscationSetupOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationSetupGetWorkspaceSize start");
static const auto aclnnObfuscationSetupGetWorkspaceSize =
    GetOpApiFunc<AclnnObfuscationSetupGetWorkspaceSizeFunc>("aclnnObfuscationSetupGetWorkspaceSize");
if (aclnnObfuscationSetupGetWorkspaceSize == nullptr) {
    return kAclnnOpApiNotFound;
}
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnObfuscationSetupGetWorkspaceSize(
        param_.fdtoClose,
        param_.dataType,
        param_.hiddenSizePerRank,
        param_.tpRank,
        0,
        0,
        param_.cmd,
        param_.threadNum,
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationSetupGetWorkspaceSize end, ret:" <<
        ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize <<
        ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);

    return ret;
}

int ObfuscationSetupOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationSetup start");
static const auto aclnnObfuscationSetup =
    GetOpApiFunc<AclnnObfuscationSetupFunc>("aclnnObfuscationSetup");
if (aclnnObfuscationSetup == nullptr) {
    return kAclnnOpApiNotFound;
}
    int ret = aclnnObfuscationSetup(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationSetup end, ret:" << ret);
    return ret;
}

} // namespace common
} // namespace atb_speed