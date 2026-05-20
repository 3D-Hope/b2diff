from backend.pipeline import UserSceneService
service = UserSceneService()
fpbpn = service.context.encoded_dataset[0]["fpbpn"]
print(fpbpn.shape)
print(fpbpn[:4])
