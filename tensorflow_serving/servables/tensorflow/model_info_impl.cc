#include "tensorflow_serving/servables/tensorflow/model_info_impl.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/contrib/session_bundle/bundle_shim.h"
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_serving/core/servable_handle.h"

namespace tensorflow {
namespace serving {

namespace {

Status ValidateModelInfoRequest(const ModelInfoRequest& request) {
  return tensorflow::Status::OK();
}

}  // namespace

Status ModelInfoImpl::GetModelInfo( ServerCore* core, const ModelInfoRequest& request, ModelInfoResponse* response) {
  TF_RETURN_IF_ERROR(ValidateModelInfoRequest(request));
  const std::vector<ServableId> available_servables = core->ListAvailableServableIds();
  std::vector<ServableId> result;

  auto models = response->mutable_servables();
  for (const auto& servable : available_servables) {
    if(request.name.is_empty()){
      std::cout << '[i] Empty ModelInfoRequest.model_name ==> returning all servables.' << std::endl;
      for (auto it = available_servables.begin() ; it != available_servables.end(); ++it) {
        models->Append(*it);
      }
    } else if(servable.name.comapre(request.model_name)){
      std::cout << '[i] ModelInfoRequest.model_name ==> returning a single servables.' << std::endl;
      models->Add(servable);
    } else {
      std::cout << '[i] Empty ModelInfoRequest.model_name ==> returning none.' << std::endl;
    }
  }
  
  return tensorflow::Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
