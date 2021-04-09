import requirement_text

with open('model_final_a6e10b.pkl', 'rb') as f:
    data = requirement_text.load(f)

print(data)

# def iterunpickle(f):
#     unpickler = pickle.Unpickler(f)
#     try:
#         while True:
#             yield unpickler.load()
#     except EOFError:
#         pass
#
# with open(r'model_final_a6e10b.pkl','rb') as pth:
#     for item in iterunpickle(pth):
#         print(item)