from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI

# load_dotenv(".env")

PROMPT="""
You are a helpful assistant. You have access to the following:

**Ensure that your adhere to the given output format without any fail and also answer only in the context. If the context given to you does not contain the information about the query, state that clearly in the response instead of answering.**
Return your answer as a valid JSON object matching this schema:
{
"title": This is the title for the particular chat query,
"response": The response answer for the query in detail. It should be a complete answer to the user's query, including all relevant information from the retrieved nodes. If the response is not clear or ambiguous, state that clearly in the response.
"citation": only the document id and the table id if the they are relevant for the response note it should not be image id,
"image": only the image id if the image is relevant for the response
}
Do not include any extra text, markdown, or formatting outside the JSON.

1. **User Query**: {user_query}
2. **Retrieved nodes**: You will receive the uuid along with each relevant node. These nodes can be text nodes or image nodes. Both contain textual information only. The only difference would be that for text nodes, you will recieve a "document UUID" and for image nodes, you will get "image UUID". Find the nodes in the following text:
{relevant_docs}.
3. **Chat History**: When answering questions, you may refer to the provided chat history {chat_history} and do not infer or assume anything that is not clearly supported by it.

**Instructions:**
- Analyze the user's query.
- Examine the text carefully.
- Try to find correlations between the nodes.
- If needed, state what is ambiguous or unclear.

(***I have provided an example response. This is only meant to help you with the output format. DONOT use this example as your response***)
Example response:
---
```json
{
"title": "Feature extraction procedure of atom 2 in the Butyramide molecule"
"response" : "The feature extraction procedure of atom 2 in the Butyramide molecule considering interested radius of 1. In particular, this algorithm involves three stages: 1. Initial Stage: Each atom ......"
"citation" : ["550e8400-e29b-41d4-a716-446655440000","520d5612-e29b-41d4-a716-434621645193"]
"image" : "f81d4fae-7dec-11d0-a765-00a0c91e6bf"
}
```
---
Some clarifications:-
citations are meant to be a list of UUIDs of all the relevant **text nodes only** and image should be the UUID of the highest relevant **image node only**, if any.
always follow the ouput structure in form of json as given in the example response
greet user if user name is present
"""

llm = OpenAI(model="gpt-3.5-turbo")
desc = OpenAI(model="gpt-4o")
def generate_response(docs, query, chad_history):

    if not docs:
            general_prompt = f"""
    You are a helpful AI assistant having a conversation with a user.

    Respond to the following question as best as you can:

    User: {query}
    Chat History: {chad_history if chad_history else "None"}
    Return your response in the following JSON format:
    {{
    "title": "A suitable title for the query",
    "response": "The answer in detail.",
    "citation": [],
    "image": null
    }}
            """
            resp = llm.complete(general_prompt)
            resp = str(resp).removeprefix("```json").removesuffix("```").strip()
            return resp
    else: 
        relevant_docs = []
        for doc in docs:
            if doc.metadata["type"]=="text":
                relevant_docs.append("document_id : {0} \n\n {1}".format(doc.node_id, doc.text))
            elif doc.metadata["type"]=="image":
                relevant_docs.append("image_id : {0} \n\n {1}".format(doc.metadata["image_uuid"], doc.text))
            else:
                relevant_docs.append("table_id: {0} \n\n {1}".format(doc.node_id, doc.text))
        print(relevant_docs, "relevant_docs")

        prompt_template = PromptTemplate(PROMPT)
        qa_prompt = prompt_template.format(relevant_docs=relevant_docs, user_query=query, chad_history=chad_history)
        resp = desc.complete(qa_prompt)
        resp = str(resp).removeprefix("```json").removesuffix('```')
        # json_resp = json.dumps(resp)
        return resp
    
if __name__ == "__main__":
    print(generate_response(["my name is gagan","i am from hubli"], "where is gagan from",None))
