//-*-C++-*-

#ifndef ANDROMEDA_SUBJECTS_DOCUMENT_DOC_NORMALISATION_H_
#define ANDROMEDA_SUBJECTS_DOCUMENT_DOC_NORMALISATION_H_

namespace andromeda
{

  template<typename doc_type>
  class doc_normalisation: public base_types
  {
    const static inline std::set<std::string> is_ignored = {"page-header", "page-footer"};

    const static inline std::set<std::string> is_text = {
      "title", "subtitle-level-1", "paragraph", "list-item",
      "footnote", "caption",
      "formula", "equation"
    };

    const static inline std::set<std::string> is_table = {"table"};
    const static inline std::set<std::string> is_figure = {"figure"};

    const static inline std::set<std::string> is_page_header = {"page-header"};
    const static inline std::set<std::string> is_page_footer = {"page-footer"};
    
  public:

    doc_normalisation(doc_type& doc);
    
    void execute_on_pdf();

  private:

    void set_pdforder();

    void init_provs();

    void sort_provs();

    void init_items();

    void link_items();

    void resolve_paths();
    
  private:

    doc_type& doc;
  };

  template<typename doc_type>
  doc_normalisation<doc_type>::doc_normalisation(doc_type& doc):
    doc(doc)
  {}

  template<typename doc_type>
  void doc_normalisation<doc_type>::execute_on_pdf()
  {
    LOG_S(WARNING);
    
    set_pdforder();

    init_provs();

    sort_provs();

    init_provs();
    
    init_items();

    link_items();

    resolve_paths();    
  }

  template<typename doc_type>
  void doc_normalisation<doc_type>::set_pdforder()
  {
    auto& orig = doc.orig;

    if(orig.count(doc_type::maintext_lbl)==0)
      {
        LOG_S(WARNING) << "no `main-text` identified";
        return;
      }

    auto& main_text = orig.at(doc_type::maintext_lbl);
    for(std::size_t pdforder=0; pdforder<main_text.size(); pdforder++)
      {
        main_text.at(pdforder)[doc_type::pdforder_lbl] = pdforder;
      }
  }

  template<typename doc_type>
  void doc_normalisation<doc_type>::init_provs()
  {
    std::string doc_name = doc.doc_name;
    
    auto& orig = doc.orig;
    auto& provs = doc.provs;

    provs.clear();

    std::string pdforder_lbl = doc_type::pdforder_lbl;    

    std::string maintext_name_lbl = doc_type::maintext_name_lbl;
    std::string maintext_type_lbl = doc_type::maintext_type_lbl;
    
    for(std::size_t l=0; l<orig.at(doc_type::maintext_lbl).size(); l++)
      {
	nlohmann::json item = orig.at(doc_type::maintext_lbl).at(l);

	std::string path="";
	if(item.count("$ref"))
	  {
	    path = item["$ref"].get<std::string>();
	  }
	else if(item.count("__ref"))
	  {
	    path = item["__ref"].get<std::string>();
	  }
	else
	  {
	    std::stringstream ss;
	    ss << "#/" << doc_type::maintext_lbl << "/" << l;

	    path = ss.str();
	  }
	
        ind_type pdforder = item.at(pdforder_lbl).get<ind_type>();
        ind_type maintext = l;

        std::string name = item.at(maintext_name_lbl).get<std::string>();
        std::string type = item.at(maintext_type_lbl).get<std::string>();

	auto prov = std::make_shared<prov_element>(pdforder, maintext,
						   path, name, type);

	std::vector<std::string> parts = utils::split(prov->get_path(), "/");
	assert(parts.size()>=3);

	std::string base = parts.at(1);
	std::size_t index = std::stoi(parts.at(2));
	
	if(orig.count(base) and index<orig[base].size()) // make sure that the path exists in the orig document
	  {
	    auto& ref_item = orig[base][index];

	    if(ref_item.count(doc_type::prov_lbl) and
	       ref_item[doc_type::prov_lbl].size()==1)
	      {
		prov->set(ref_item[doc_type::prov_lbl][0]);		
		provs.push_back(prov);
	      }
	    else
	      {
		LOG_S(ERROR) << "undefined prov for main-text item: "
			     << item.dump();
	      } 
	  }
	else
	  {
	    LOG_S(WARNING) << "undefined reference path in document: "
			   << item.dump();
	  }
      }
  }
  
  template<typename doc_type>
  void doc_normalisation<doc_type>::sort_provs()
  {
    doc_order sorter;
    sorter.order_maintext(doc);
  }

  template<typename doc_type>
  void doc_normalisation<doc_type>::init_items()
  {
    std::string doc_name = doc.doc_name;
    
    auto& orig = doc.orig;
    auto& provs = doc.provs;
    
    auto& texts = doc.texts;
    auto& tables = doc.tables;
    auto& figures = doc.figures;

    auto& page_headers = doc.page_headers;
    auto& page_footers = doc.page_footers;
    auto& other = doc.other;
    
    {
      texts.clear();
      tables.clear();
      figures.clear();

      page_headers.clear();
      page_footers.clear();
      other.clear();
    }

    for(uint64_t i=0; i<provs.size(); i++)
      {
	auto& prov = provs.at(i);
	
	// set a self-reference for later use after sorting ...
	{
	  std::stringstream ss;
	  ss << "#/" << doc_type::provs_lbl << "/" << i;
	  prov->set_pref(ss.str());
	}
	
        std::vector<std::string> parts = utils::split(prov->get_path(), "/");

        std::string base = parts.at(1);
        std::size_t index = std::stoi(parts.at(2));

        auto& item = orig.at(base).at(index);

        if(is_text.count(prov->get_type()))
          {
	    std::stringstream ss;
	    ss << doc_name << "#/" << doc_type::texts_lbl << "/" << texts.size();	    

	    std::string dloc = ss.str();
	    
            auto subj = std::make_shared<subject<TEXT> >(doc.doc_hash, dloc, prov);
            bool valid = subj->set_data(item);

            if(valid)
              {
                texts.push_back(subj);
              }
            else
              {
                LOG_S(WARNING) << "found invalid text: " << item.dump();
              }
          }
        else if(is_table.count(prov->get_type()))
          {
	    std::stringstream ss;
	    ss << doc_name << "#/" << doc_type::tables_lbl << "/" << tables.size();	    

	    std::string dloc = ss.str();
	    
            auto subj = std::make_shared<subject<TABLE> >(doc.doc_hash, dloc, prov);
            bool valid = subj->set_data(item);

	    tables.push_back(subj);

	    if(not valid)
	      {
                LOG_S(WARNING) << "invalid table: "<< prov->get_path();
	      }
	    else
	      {
		//LOG_S(WARNING) << "valid table: " << prov->path;
	      }
          }
        else if(is_figure.count(prov->get_type()))
          {
	    std::stringstream ss;
	    ss << doc_name << "#/" << doc_type::figures_lbl << "/" << figures.size();	    

	    std::string dloc = ss.str();
	    
            auto subj = std::make_shared<subject<FIGURE> >(doc.doc_hash, dloc, prov);
            bool valid = subj->set_data(item);

	    figures.push_back(subj);
	    
            if(not valid)
	      {
                LOG_S(WARNING) << "found figure without structure";
              }
          }
        else if(is_page_header.count(prov->get_type()))
          {
	    std::stringstream ss;
	    ss << doc_name << "#/" << doc_type::page_headers_lbl << "/" << page_headers.size();	    

	    std::string dloc = ss.str();
	    
            auto subj = std::make_shared<subject<TEXT> >(doc.doc_hash, dloc, prov);
            bool valid = subj->set_data(item);

            if(valid)
              {
                page_headers.push_back(subj);
              }
            else
              {
                LOG_S(WARNING) << "found invalid text: " << item.dump();
              }
          }	
        else if(is_page_footer.count(prov->get_type()))
          {
	    std::stringstream ss;
	    ss << doc_name << "#/" << doc_type::page_footers_lbl << "/" << page_footers.size();	    

	    std::string dloc = ss.str();
	    
            auto subj = std::make_shared<subject<TEXT> >(doc.doc_hash, dloc, prov);
            bool valid = subj->set_data(item);

            if(valid)
              {
                page_footers.push_back(subj);
              }
            else
              {
                LOG_S(WARNING) << "found invalid text: " << item.dump();
              }
          }	
        else
          {
	    prov->set_ignored(true);
            if(not is_ignored.count(prov->get_type()))
              {
                LOG_S(WARNING) << "found new `other` type: " << prov->get_type();
              }

	    std::stringstream ss;
	    ss << doc_name << "#/" << doc_type::meta_lbl << "/" << texts.size();	    

	    std::string dloc = ss.str();
	    
            auto subj = std::make_shared<subject<TEXT> >(doc.doc_hash, dloc, prov);
            bool valid = subj->set_data(item);

            if(valid)
              {
                other.push_back(subj);
              }
            else
              {
                LOG_S(WARNING) << "found invalid text: " << item.dump();
              }
          }
      }
  }

  template<typename doc_type>
  void doc_normalisation<doc_type>::link_items()
  {
    {
      doc_captions linker;
      linker.find_and_link_captions(doc);
    }

    {
      doc_maintext linker;

      linker.filter_maintext(doc);
      linker.concatenate_maintext(doc);
    }
  }

  template<typename doc_type>
  void doc_normalisation<doc_type>::resolve_paths()
  {
    auto& texts = doc.texts;
    auto& tables = doc.tables;
    auto& figures = doc.figures;    
    
    for(index_type l=0; l<texts.size(); l++)
      {
	for(auto& prov:texts.at(l)->provs)
	  {
	    std::stringstream ss;
	    ss << "#/" << doc_type::texts_lbl << "/" << l;

	    prov->set_path(ss.str());
	  }
      }

    for(index_type l=0; l<tables.size(); l++)
      {
	for(auto& prov:tables.at(l)->provs)
	  {
	    std::stringstream ss;
	    ss << "#/" << doc_type::tables_lbl << "/" << l;

	    prov->set_path(ss.str());
	  }

	for(index_type k=0; k<tables.at(l)->captions.size(); k++)
	  {
	    for(auto& prov:tables.at(l)->captions.at(k)->provs)
	      {
		std::stringstream ss;
		ss << "#/"
		   << doc_type::tables_lbl << "/" << l << "/"
		   << doc_type::captions_lbl << "/" << k;
		
		prov->set_path(ss.str());
	      }
	  }
      }

    for(index_type l=0; l<figures.size(); l++)
      {
	for(auto& prov:figures.at(l)->provs)
	  {
	    std::stringstream ss;
	    ss << "#/" << doc_type::figures_lbl << "/" << l;

	    prov->set_path(ss.str());
	  }

	for(index_type k=0; k<figures.at(l)->captions.size(); k++)
	  {
	    for(auto& prov:figures.at(l)->captions.at(k)->provs)
	      {
		std::stringstream ss;
		ss << "#/"
		   << doc_type::figures_lbl << "/" << l << "/"
		   << doc_type::captions_lbl << "/" << k;
		
		prov->set_path(ss.str());
	      }
	  }	
      }        
  }
  
}

#endif
